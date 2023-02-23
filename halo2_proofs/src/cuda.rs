#[cfg(test)]
mod tests {
    use crypto_cuda::{
        create_kernel_with_params, cuda_get_device_count, cuda_init, kernel::CudaBlock,
        make_params, CudaContext, CudaDevice, CudaModule, CudaResult, CudaStream, DeviceMemory,
        HostMemory,
    };
    use halo2curves::bn256::Fq;
    use std::path::PathBuf;

    const BUF_LEN: usize = 1 << 10;
    const BUCKET_LEN: usize = 1 << 8;

    #[test]
    fn test_ff_add() -> CudaResult<()> {
        cuda_init()?;
        let dev = CudaDevice::new(0)?;
        let ctx = CudaContext::new(dev)?;

        let lhs_value = Fq::from_raw([
            0x1212121212121212,
            0x1212121212121212,
            0x1212121212121212,
            0x1212121212121212,
        ]);

        let rhs_value = Fq::from_raw([
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
        ]);

        let mut lhs_host = HostMemory::<Fq>::new(1)?;
        let mut rhs_host = HostMemory::<Fq>::new(1)?;
        let mut out_host = HostMemory::<Fq>::new(1)?;

        let lhs = DeviceMemory::<Fq>::new(&ctx, 1)?;
        let rhs = DeviceMemory::<Fq>::new(&ctx, 1)?;
        let out = DeviceMemory::<Fq>::new(&ctx, 1)?;

        lhs_host[0] = lhs_value;
        rhs_host[0] = rhs_value;
        lhs.read_from(&lhs_host, 1)?;
        rhs.read_from(&rhs_host, 1)?;

        let module = CudaModule::new("ff.ptx")?;
        let f = module.get_func("ff_add")?;

        let stream = CudaStream::new_with_context(&ctx)?;
        let params = make_params!(lhs.get_inner(), rhs.get_inner(), out.get_inner());
        let kernel = create_kernel_with_params!(f, <<<1, 1, 0>>>(params));

        stream.launch(&kernel)?;
        stream.sync()?;

        out.write_to(&mut out_host, 1)?;
        assert_eq!(out_host[0], lhs_value.add(&rhs_value));
        Ok(())
    }

    #[test]
    fn test_ff_sub() -> CudaResult<()> {
        cuda_init()?;
        let dev = CudaDevice::new(0)?;
        let ctx = CudaContext::new(dev)?;

        let lhs_value = Fq::from_raw([
            0x1212121212121212,
            0x1212121212121212,
            0x1212121212121212,
            0x1212121212121212,
        ]);

        let rhs_value = Fq::from_raw([
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
            0xf2f2f2f2f2f2f2f2,
        ]);

        let mut lhs_host = HostMemory::<Fq>::new(1)?;
        let mut rhs_host = HostMemory::<Fq>::new(1)?;
        let mut out_host = HostMemory::<Fq>::new(1)?;

        let lhs = DeviceMemory::<Fq>::new(&ctx, 1)?;
        let rhs = DeviceMemory::<Fq>::new(&ctx, 1)?;
        let out = DeviceMemory::<Fq>::new(&ctx, 1)?;

        lhs_host[0] = lhs_value;
        rhs_host[0] = rhs_value;
        lhs.read_from(&lhs_host, 1)?;
        rhs.read_from(&rhs_host, 1)?;

        let module = CudaModule::new("ff.ptx")?;
        let f = module.get_func("ff_sub")?;

        let stream = CudaStream::new_with_context(&ctx)?;
        let params = make_params!(lhs.get_inner(), rhs.get_inner(), out.get_inner());
        let kernel = create_kernel_with_params!(f, <<<1, 1, 0>>>(params));

        stream.launch(&kernel)?;
        stream.sync()?;

        out.write_to(&mut out_host, 1)?;
        assert_eq!(out_host[0], lhs_value.sub(&rhs_value));
        Ok(())
    }

    #[test]
    fn test_kernel() -> CudaResult<()> {
        cuda_init()?;

        let n = cuda_get_device_count()?;

        let mut devs = Vec::new();
        let mut ctxs = Vec::new();

        for i in 0..n {
            let dev = CudaDevice::new(i)?;
            devs.push(dev);
            ctxs.push(CudaContext::new(dev)?);
        }

        let bases = DeviceMemory::<u32>::new(&ctxs[0], BUF_LEN)?;
        let scalars = DeviceMemory::<u32>::new(&ctxs[0], BUF_LEN)?;
        let buckets = DeviceMemory::<u32>::new(&ctxs[0], BUCKET_LEN)?;

        let mut bases_host = HostMemory::<u32>::new(BUF_LEN)?;
        let mut scalars_host = HostMemory::<u32>::new(BUF_LEN)?;
        let mut buckets_host = HostMemory::<u32>::new(BUCKET_LEN)?;

        for i in 0..BUF_LEN {
            bases_host[i] = 1u32;
            scalars_host[i] = (i % BUCKET_LEN) as u32;
        }

        bases.read_from(&bases_host, BUF_LEN)?;
        scalars.read_from(&scalars_host, BUF_LEN)?;

        let module = CudaModule::new(
            [env!("OUT_DIR"), "msm.ptx"]
                .iter()
                .collect::<PathBuf>()
                .to_str()
                .unwrap(),
        )?;

        let f = module.get_func("make_buckets")?;

        let stream = CudaStream::new_with_context(&ctxs[0])?;

        let params = make_params!(
            bases.get_inner(),
            scalars.get_inner(),
            buckets.get_inner(),
            BUF_LEN as u32
        );
        let kernel = create_kernel_with_params!(f, <<<1, BUCKET_LEN as u32, 0>>>(params));

        stream.launch(&kernel)?;
        stream.sync()?;

        buckets.write_to(&mut buckets_host, BUCKET_LEN)?;
        Ok(())
    }
}
