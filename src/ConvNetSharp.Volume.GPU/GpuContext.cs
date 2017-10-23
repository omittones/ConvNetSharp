using System;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.CudaDNN;
using ManagedCuda.CudaBlas;

namespace ConvNetSharp.Volume.GPU
{
    public class GpuContext
    {
        private static readonly Lazy<GpuContext> DefaultContextLazy = new Lazy<GpuContext>(() => new GpuContext(0));
        private CudaContext _cudaContext;

        private CudaStream _defaultStream;

        public GpuContext(int deviceId = 0)
        {
            this.CudaContext = new CudaContext(deviceId, true);

            var props = this.CudaContext.GetDeviceInfo();
            this.DefaultBlockCount = props.MultiProcessorCount * 32;
            this.DefaultThreadsPerBlock = props.MaxThreadsPerBlock;
            this.WarpSize = props.WarpSize;

            this.DefaultStream = new CudaStream();            
            this.CudnnContext = new CudaDNNContextEx();
            this.CublasContext = new CudaBlas(this.DefaultStream.Stream, PointerMode.Device, AtomicsMode.NotAllowed);
        }

        public CudaContext CudaContext
        {
            get { return this._cudaContext; }
            private set { this._cudaContext = value; }
        }

        public CudaDNNContextEx CudnnContext { get; }

        public CudaBlas CublasContext { get; }

        public static GpuContext Default => DefaultContextLazy.Value;

        public int DefaultBlockCount { get; }

        public int DefaultThreadsPerBlock { get; }

        public int WarpSize { get; }

        public CudaStream DefaultStream
        {
            get { return this._defaultStream; }
            set { this._defaultStream = value; }
        }

        public void Dispose()
        {
            Dispose(true);
        }

        public virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (disposing)
            {
                Dispose(ref this._defaultStream);
                Dispose(ref this._cudaContext);
            }
        }

        public void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try
                {
                    field.Dispose();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }
                field = null;
            }
        }

        ~GpuContext()
        {
            Dispose(false);
        }
    }
}