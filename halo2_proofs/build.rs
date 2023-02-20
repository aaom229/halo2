use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;

fn main(){
    let nvcc = which::which("nvcc");
    if nvcc.is_ok() {
        let out_dir = env::var("OUT_DIR").unwrap();
        let out_file = {
            let mut path = PathBuf::from_str(&out_dir).unwrap(); 
            path.push("msm.ptx");
            path
        };
        print!("cargo:warning={:?}", out_dir);
        let mut handle = Command::new(nvcc.unwrap())
            .arg("--ptx")
            .arg("src/cuda/msm.cu")
            .arg("-o")
            .arg(out_file)
            .spawn().unwrap();
        handle.wait().unwrap();
    }

    println!("cargo:rerun-if-changed=src/cuda/msm.cu");
}
