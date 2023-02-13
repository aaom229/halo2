fn main() {
    /* Link CUDA Driver API (libcuda.so) */

    println!("cargo:rustc-link-search=native=/opt/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=cuda");

    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
