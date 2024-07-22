#[cfg(not(feature = "gpu"))]
fn main() {}

#[cfg(feature = "gpu")]
fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=src/");

    // Compile drillx
    cc::Build::new()
        .cuda(true)
        .include("cuda/equix/include")
        .include("cuda/equix/src")
        .include("cuda/hashx/include")
        .include("cuda/hashx/src")
        .file("cuda/drillx.cu")
        .file("cuda/equix/src/context.cu")
        .file("cuda/equix/src/equix.cu")
        .file("cuda/equix/src/solver.cu")
        .file("cuda/hashx/src/blake2.cu")
        .file("cuda/hashx/src/compiler.cu")
        .file("cuda/hashx/src/context.cu")
        .file("cuda/hashx/src/hashx.cu")
        .file("cuda/hashx/src/program.cu")
        .file("cuda/hashx/src/program_exec.cu")
        .file("cuda/hashx/src/siphash.cu")
        .file("cuda/hashx/src/siphash_rng.cu")
        .flag("-cudart=static")
        .flag("-diag-suppress=174")
        .flag("-gencode=arch=compute_86,code=sm_86") // Optimize for RTX 4090
        .flag("-gencode=arch=compute_86,code=compute_86") // PTX for future compatibility
        .compile("drillx.a");

    // Add link directory
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
