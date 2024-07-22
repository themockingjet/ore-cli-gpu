# ORE CLI

A command line interface for ORE cryptocurrency mining.

## Install

To install the CLI, use [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html):

```sh
cargo build --features gpu
```

## Build

To build the codebase from scratch, checkout the repo and use cargo to build:

```sh
cargo run --release --features gpu
```

## Help

You can use the `-h` flag on any command to pull up a help menu with documentation:

```sh
./target/release/ore -h
```
