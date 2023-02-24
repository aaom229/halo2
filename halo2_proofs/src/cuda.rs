#[cfg(test)]
mod tests {
    use crypto_cuda::{
        create_kernel_with_params, cuda_get_device_count, cuda_init, kernel::CudaBlock,
        make_params, CudaContext, CudaDevice, CudaModule, CudaResult, CudaStream, DeviceMemory,
        HostMemory,
    };
    use ff::Field;
    use group::Group;
    use halo2curves::bn256::{Bn256, Fq, G1};
    use rand_core::{OsRng, impls::next_u64_via_fill};
    use std::path::PathBuf;

    macro_rules! launch {
        ($ptx: literal, $func: literal, $($name: ident : [$type: ty; $size: literal] => $values: expr),* $(,)?) => {
            cuda_init()?;
            let dev = CudaDevice::new(0)?;
            let ctx = CudaContext::new(dev)?;

            $(
                let mut $name: ([$type; $size], HostMemory<$type>, DeviceMemory<$type>) = (
                    $values,
                    HostMemory::<$type>::new($size)?,
                    DeviceMemory::<$type>::new(&ctx, $size)?
                );
                $name.2.read_from(&$name.0, $size)?;
            )*
            let module = CudaModule::new($ptx)?;
            let f = module.get_func($func)?;

            let stream = CudaStream::new_with_context(&ctx)?;
            let params = make_params!($($name.2.get_inner()),*);
            let kernel = create_kernel_with_params!(f, <<<1,1,0>>>(params));
            stream.launch(&kernel)?;
            stream.sync()?;

            $(
                $name.2.write_to(&mut $name.1, $size)?;
            )*
        };
    }

    #[test]
    fn test_g1_dbl() -> CudaResult<()> {
        launch!("ff.ptx", "g1_dbl",
            a: [G1; 1] => [
                G1::random(OsRng)
            ],
            out: [G1; 1] => [
                G1::identity()
            ],
        );
        assert_eq!(out.1[0], a.1[0].double());
        Ok(())
    }

    #[test]
    fn test_ff_mul() -> CudaResult<()> {
        launch!("ff.ptx", "ff_mul", 
            a: [Fq;1] => [
                Fq::random(OsRng)
            ],
            b: [Fq;1] => [
                Fq::random(OsRng)
            ],
            out: [Fq;1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], a.1[0].mul(&b.1[0]));
        Ok(())
    }

    #[test]
    fn test_ff_square() -> CudaResult<()> {
        launch!("ff.ptx", "ff_square",
            a: [Fq;1] => [
                Fq::random(OsRng)
            ],
            out: [Fq;1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], a.1[0].square());
        Ok(())
    }

    #[test]
    fn test_ff_montgomery_reduce() -> CudaResult<()> {
        launch!("ff.ptx", "ff_montgomery_reduce", 
            a: [u64; 8] => [
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
                next_u64_via_fill(&mut OsRng),
            ], 
            out: [Fq; 1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], Fq::montgomery_reduce(
            a.0[0],
            a.0[1],
            a.0[2],
            a.0[3],
            a.0[4],
            a.0[5],
            a.0[6],
            a.0[7],
        ));
        Ok(())
    }

    #[test]
    fn test_ff_add() -> CudaResult<()> {
        launch!("ff.ptx", "ff_add", 
            a: [Fq;1] => [
                Fq::random(OsRng)
            ],
            b: [Fq;1] => [
                Fq::random(OsRng)
            ],
            out: [Fq;1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], a.1[0].add(&b.1[0]));
        Ok(())
    }

    #[test]
    fn test_ff_sub() -> CudaResult<()> {
        launch!("ff.ptx", "ff_sub", 
            a: [Fq;1] => [
                Fq::random(OsRng)
            ],
            b: [Fq;1] => [
                Fq::random(OsRng)
            ],
            out: [Fq;1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], a.1[0].sub(&b.1[0]));
        Ok(())
    }
}
