#[cfg(test)]
mod tests {
    use crypto_cuda::{
        create_kernel_with_params, cuda_get_device_count, cuda_init, kernel::CudaBlock,
        make_params, CudaContext, CudaDevice, CudaModule, CudaResult, CudaStream, DeviceMemory,
        HostMemory,
    };
    use ff::{Field, PrimeField};
    use group::Group;
    use halo2curves::bn256::{Bn256, Fq, G1};
    use rand_core::{impls::next_u64_via_fill, OsRng};
    use std::path::PathBuf;

    macro_rules! launch {
        ($ptx: literal, $func: literal, <<<$blocks: literal, $theards: literal, $mem: literal >>>, $($name: ident : [$type: ty; $size: expr] => $values: expr),* $(,)?) => {
            cuda_init().unwrap();
            let dev = CudaDevice::new(0).unwrap();
            let ctx = CudaContext::new(dev).unwrap();

            $(
                let mut $name: ([$type; $size], HostMemory<$type>, DeviceMemory<$type>) = (
                    $values,
                    HostMemory::<$type>::new($size).unwrap(),
                    DeviceMemory::<$type>::new(&ctx, $size).unwrap()
                );
                $name.2.read_from(&$name.0, $size).unwrap();
            )*
            let module = CudaModule::new($ptx).unwrap();
            let f = module.get_func($func).unwrap();

            let stream = CudaStream::new_with_context(&ctx).unwrap();
            let params = make_params!($($name.2.get_inner()),*);
            let kernel = create_kernel_with_params!(f, <<<$blocks,$theards,$mem>>>(params));
            stream.launch(&kernel).unwrap();
            stream.sync().unwrap();

            $(
                $name.2.write_to(&mut $name.1, $size).unwrap();
            )*
        };
    }

    #[test]
    fn test_pippenger_make_buckets() -> CudaResult<()> {
        macro_rules! inc {
            ($i: literal) => {
                {
                    let mut tmp: [u8; 32] = [0; 32];
                    for i in 0..32 {
                        tmp[i] = i as u8 + $i * 32;
                    }
                    tmp
                }
            };
        }
        launch!("ff.ptx", "pippenger_make_buckets", <<<32, 256, 0>>>,
            cfg: [u32; 4] => [
                8, 0, 32, 1
            ],
            bases: [G1; 8] => [G1::generator(); 8],
            scalars: [[u8; 32]; 8] => [
                inc!(0),
                inc!(1),
                inc!(2),
                inc!(3),
                inc!(4),
                inc!(5),
                inc!(6),
                inc!(7),
            ],
            buckets: [G1; 256*32] => [G1::identity(); 256*32],
        );
        // segment0: []
        let res = buckets.1.as_slice();

        for segment_idx in 1..32 {
            for bucket_idx in 0..256 {
                let idx = [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0].map(|e| e + segment_idx);
                if idx.contains(&bucket_idx) {
                    assert_eq!(res[segment_idx * 256 + bucket_idx], G1::generator());
                } else {
                    assert_eq!(res[segment_idx * 256 + bucket_idx], G1::identity());
                }
            }
            break;
        }
        Ok(())
    }

    #[test]
    fn test_g1_add() -> CudaResult<()> {
        launch!("ff.ptx", "g1_add", <<<1, 1, 0>>>,
            a: [G1; 1] => [
                G1::random(OsRng)
            ],
            b: [G1; 1] => [
                G1::random(OsRng)
            ],
            out: [G1; 1] => [
                G1::identity()
            ],
        );
        assert_eq!(out.1[0], a.1[0] + b.1[0]);
        Ok(())
    }

    #[test]
    fn test_g1_dbl() -> CudaResult<()> {
        launch!("ff.ptx", "g1_dbl",<<<1, 1, 0>>>,
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
        launch!("ff.ptx", "ff_mul", <<<1, 1, 0>>>,
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
        launch!("ff.ptx", "ff_square",<<<1, 1, 0>>>,
            a: [Fq;1] => [
                Fq::random(OsRng)
            ],
            out: [Fq;1] => [Fq::zero()],
        );
        assert_eq!(out.1[0], a.1[0].square());
        Ok(())
    }

    #[test]
    fn test_ff_add() -> CudaResult<()> {
        launch!("ff.ptx", "ff_add", <<<1, 1, 0>>>,
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
        launch!("ff.ptx", "ff_sub", <<<1, 1, 0>>>,
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
