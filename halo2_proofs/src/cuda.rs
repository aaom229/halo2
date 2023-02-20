#[cfg(test)]
mod tests {
    use crypto_cuda::{
        create_kernel_with_params, cuda_get_device_count, cuda_init, kernel::CudaBlock,
        make_params, CudaContext, CudaDevice, CudaModule, CudaResult, CudaStream, DeviceMemory,
        HostMemory,
    };
    use std::path::PathBuf;

    const BUF_LEN: usize = 1 << 10;
    const BUCKET_LEN: usize = 1 << 8;

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
