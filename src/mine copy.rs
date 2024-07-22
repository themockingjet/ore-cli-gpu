use std::{sync::Arc, time::Instant};

use colored::*;

use drillx::{
    equix::{self},
    Hash, Solution,
};

#[cfg(feature = "gpu")]
extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
    pub fn solve_all_stages(hashes: *const u64, out: *mut u8, sols: *mut u32);
}

use ore_api::{
    consts::{BUS_ADDRESSES, BUS_COUNT, EPOCH_DURATION},
    state::{Config, Proof},
};

use rand::Rng;
use solana_program::pubkey::Pubkey;
use solana_rpc_client::spinner;
use solana_sdk::signer::Signer;

use crate::{
    args::MineArgs,
    send_and_confirm::ComputeBudget,
    utils::{amount_u64_to_string, get_clock, get_config, get_proof_with_authority},
    Miner,
};

use std::sync::Mutex;
use rayon::prelude::*;

impl Miner {
    pub async fn mine(&self, args: MineArgs) {
        // Register, if needed.
        let signer = self.signer();
        self.open().await;

        // Check num threads
        self.check_num_cores(args.threads);

        // Start mining loop
        loop {
            // Fetch proof
            let proof = get_proof_with_authority(&self.rpc_client, signer.pubkey()).await;
            println!(
                "\nStake balance: {} ORE",
                amount_u64_to_string(proof.balance)
            );

            // Calc cutoff time
            let cutoff_time = self.get_cutoff(proof, args.buffer_time).await;

            // Run drillx
            let solution = if cfg!(feature = "gpu") {
                Self::find_hash_gpu(proof, cutoff_time).await
            } else {
                Self::find_hash_par(proof, cutoff_time, args.threads).await
            };

            // Submit most difficult hash
            let config = get_config(&self.rpc_client).await;
            let mut compute_budget = 500_000;
            let mut ixs = vec![];
            if self.should_reset(config).await {
                compute_budget += 100_000;
                ixs.push(ore_api::instruction::reset(signer.pubkey()));
            }
            if self.should_crown(config, proof).await {
                compute_budget += 250_000;
                ixs.push(ore_api::instruction::crown(
                    signer.pubkey(),
                    config.top_staker,
                ))
            }
            ixs.push(ore_api::instruction::mine(
                signer.pubkey(),
                signer.pubkey(),
                find_bus(),
                solution,
            ));
            self.send_and_confirm(&ixs, ComputeBudget::Fixed(compute_budget), false)
                .await
                .ok();
        }
    }

    async fn find_hash_par(proof: Proof, cutoff_time: u64, threads: u64) -> Solution {
        // Dispatch job to each thread
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message("Mining...");

        let difficulties = Arc::new(Mutex::new(vec![0; threads as usize]));

        let handles: Vec<_> = (0..threads)
        .into_par_iter()
        .map(|i| {
            std::thread::spawn({
                let proof = proof.clone();
                let progress_bar = progress_bar.clone();
                let mut memory = equix::SolverMemory::new();
                let difficulties = difficulties.clone();
                move || {
                    let timer = Instant::now();
                    let mut nonce = u64::MAX.saturating_div(threads).saturating_mul(i);
                    let mut best_nonce = nonce;
                    let mut best_difficulty = 0;
                    let mut best_hash = Hash::default();
                    loop {
                        // Create hash
                        if let Ok(hx) = drillx::hash_with_memory(
                            &mut memory,
                            &proof.challenge,
                            &nonce.to_le_bytes(),
                        ) {
                            let difficulty = hx.difficulty();
                            if difficulty.gt(&best_difficulty) {
                                best_nonce = nonce;
                                best_difficulty = difficulty;
                                best_hash = hx;
                                let mut difficulties = difficulties.lock().unwrap();
                                difficulties[i as usize] = best_difficulty;
                            }
                        }

                        // Exit if time has elapsed
                        if nonce % 100 == 0 {
                            if timer.elapsed().as_secs().ge(&cutoff_time) {
                                if best_difficulty.gt(&ore_api::consts::MIN_DIFFICULTY) {
                                    // Mine until min difficulty has been met
                                    break;
                                }
                            } else if i == 0 {
                                let difficulties = difficulties.lock().unwrap();
                                let message = difficulties.iter().enumerate()
                                    .map(|(i, &diff)| format!("T{}: {}", i, diff))
                                    .collect::<Vec<_>>().join("\n");
                                progress_bar.set_message(format!(
                                    "Mining... ({} sec remaining) [{}]",
                                    cutoff_time.saturating_sub(timer.elapsed().as_secs()),
                                    message,
                                ));
                            }
                        }

                        // Increment nonce
                        nonce = nonce.saturating_add(999);
                    }

                    // Return the best nonce
                    (best_nonce, best_difficulty, best_hash)
                }
            })
        })
        .collect();

        // Join handles and return best nonce
        let mut best_nonce = 0;
        let mut best_difficulty = 0;
        let mut best_hash = Hash::default();
        for h in handles {
            if let Ok((nonce, difficulty, hash)) = h.join() {
                if difficulty > best_difficulty {
                    best_difficulty = difficulty;
                    best_nonce = nonce;
                    best_hash = hash;
                }
            }
        }

        // Update log
        progress_bar.finish_with_message(format!(
            "Best hash: {} (difficulty: {})",
            bs58::encode(best_hash.h).into_string(),
            best_difficulty
        ));

        Solution::new(best_hash.d, best_nonce.to_le_bytes())
    }


    #[cfg(feature = "gpu")]
    async fn find_hash_gpu(proof: Proof, cutoff_time: u64) -> Solution {
        let threads = num_cpus::get();
    
        // Progress Bar
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message("Mining with GPU...");
    
        // Initialize
        let timer = Instant::now();
        let proof = proof.clone();
    
        // Constants
        const INDEX_SPACE: usize = 65536;
        let x_batch_size = unsafe { BATCH_SIZE };
    
        // Pre-allocate memory
        let mut hashes = vec![0u64; x_batch_size as usize * INDEX_SPACE];
        let mut digest = [0u8; 16];
        let mut sols = [0u8; 4];
    
        // nonce
        let mut x_nonce = 0u64;
    
        // Final results
        let mut xbest_nonce = 0;
        let mut xbest_difficulty = 0;
        let mut xbest_hash = Hash::default();
    
        loop {
            let total_nonces = x_batch_size as usize * INDEX_SPACE;
    
            unsafe {
                // use GPU for hashing
                hash(
                    proof.challenge.as_ptr(),
                    &x_nonce as *const u64 as *const u8,
                    hashes.as_mut_ptr() as *mut u64,
                );
    
                // use CPU for solving
                let chunk_size = x_batch_size as usize / threads;
                let handles = (0..threads).into_par_iter().map(|i| {
                    let start = i * chunk_size;
                    let end = if i + 1 == threads { x_batch_size as usize } else { start + chunk_size };
            
                    let mut best_nonce = 0;
                    let mut best_difficulty = 0;
                    let mut best_hash = Hash::default();
            
                    for i in start..end {
                        let batch_start = hashes.as_ptr().add(i * INDEX_SPACE);
                        solve_all_stages(
                            batch_start,
                            digest.as_mut_ptr(),
                            sols.as_mut_ptr() as *mut u32,
                        );
                        if u32::from_le_bytes(sols) > 0 {
                            let solution = Solution::new(digest, (x_nonce + i as u64).to_le_bytes());
                            let difficulty = solution.to_hash().difficulty();
                            if  solution.is_valid(&proof.challenge) && difficulty > best_difficulty {
                                best_nonce = u64::from_le_bytes(solution.n);
                                best_difficulty = difficulty;
                                best_hash = solution.to_hash();
                            }
                        }
                    }
            
                    (best_nonce, best_difficulty, best_hash)
                }).reduce_with(|a, b| {
                    if a.1 > b.1 { a } else { b }
                }).unwrap();
            
    
                if handles.1 > xbest_difficulty {
                    xbest_nonce = handles.0;
                    xbest_difficulty = handles.1;
                    xbest_hash = handles.2;
                }
    
                // Increment nonce for next batch
                x_nonce += total_nonces as u64;
    
                // Update progress bar less frequently
                if x_nonce % (total_nonces as u64 * 100) == 0 {
                    let elapsed = timer.elapsed().as_secs();
                    let hashes_per_second = (x_nonce as f64) / elapsed as f64;
                    progress_bar.set_message(format!(
                        "Mining with GPU... (Best difficulty: {}, Hashes/s: {:.2}, Time: {}s)",
                        xbest_difficulty,
                        hashes_per_second,
                        elapsed
                    ));
                }
    
                // Exit if time has elapsed or sufficient difficulty is reached
                if timer.elapsed().as_secs() >= cutoff_time && xbest_difficulty > ore_api::consts::MIN_DIFFICULTY {
                    break;
                }
            }
        }
    
        // Update log
        progress_bar.finish_with_message(format!(
            "Best hash: {} (difficulty: {})",
            bs58::encode(xbest_hash.h).into_string(),
            xbest_difficulty
        ));
    
        Solution::new(xbest_hash.d, xbest_nonce.to_le_bytes())
    }


    pub fn check_num_cores(&self, threads: u64) {
        // Check num threads
        let num_cores = num_cpus::get() as u64;
        if threads.gt(&num_cores) {
            println!(
                "{} Number of threads ({}) exceeds available cores ({})",
                "WARNING".bold().yellow(),
                threads,
                num_cores
            );
        }
    }

    async fn should_crown(&self, config: Config, proof: Proof) -> bool {
        proof.balance.gt(&config.max_stake)
    }

    async fn should_reset(&self, config: Config) -> bool {
        let clock = get_clock(&self.rpc_client).await;
        config
            .last_reset_at
            .saturating_add(EPOCH_DURATION)
            .saturating_sub(5) // Buffer
            .le(&clock.unix_timestamp)
    }

    async fn get_cutoff(&self, proof: Proof, buffer_time: u64) -> u64 {
        let clock = get_clock(&self.rpc_client).await;
        proof
            .last_hash_at
            .saturating_add(60)
            .saturating_sub(buffer_time as i64)
            .saturating_sub(clock.unix_timestamp)
            .max(0) as u64
    }
}

// TODO Pick a better strategy (avoid draining bus)
fn find_bus() -> Pubkey {
    let i = rand::thread_rng().gen_range(0..BUS_COUNT);
    BUS_ADDRESSES[i]
}
