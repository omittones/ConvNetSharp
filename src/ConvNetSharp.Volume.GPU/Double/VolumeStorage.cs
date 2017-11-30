using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Double
{
    public unsafe class VolumeStorage : VolumeStorage<double>, IDisposable, IVolumeStorage<double>
    {
        private readonly VolumeStorage _originalStorage;
        private readonly CudaHostMemoryRegion _hostPointer;
        private bool _allocatedOnDevice;
        private bool _disposed;
        public DataLocation Location { get; set; }
        public CudaDeviceVariable<double> DeviceBuffer { get; private set; }
        public double* HostBuffer => (double*)this._hostPointer.Start;
        public GpuContext Context { get; }

        public VolumeStorage(Shape shape, GpuContext context, long length = -1) : base(shape)
        {
            this.Context = context;

            // Take care of unkown dimension
            if (length != -1)
            {
                this.Shape.GuessUnkownDimension(length);
            }

            // Host 
            this._hostPointer = InitializeSharedMemory(this.Shape.TotalLength);

            this._originalStorage = null;
        }
        
        public VolumeStorage(double[] array, Shape shape, GpuContext context) : this(shape, context, array.Length)
        {
            this.Context = context;

            if (this.Shape.TotalLength != array.Length)
            {
                throw new ArgumentException("Wrong dimensions");
            }

            // Fill host buffer
            for (var i = 0; i < array.Length; i++)
            {
                this.HostBuffer[i] = array[i];
            }

            this.Location = DataLocation.Host;
        }

        public VolumeStorage(VolumeStorage storage, Shape newShape) : base(newShape)
        {
            if (storage == null)
                throw new ArgumentNullException(nameof(storage));

            if (storage._hostPointer == null)
                throw new ArgumentException();

            this.Shape = newShape;
            this.Context = storage.Context;
            this._originalStorage = storage;
            this._hostPointer = storage._hostPointer;
            this._allocatedOnDevice = storage._allocatedOnDevice;

            storage.CopyToDevice();

            this.Location = DataLocation.Device;
            this.DeviceBuffer = new CudaDeviceVariable<double>(storage.DeviceBuffer.DevicePointer);
        }

        public override void Set(double[] values)
        {
            throw new NotImplementedException();
        }

        public long GpuMemory => this.Shape.TotalLength * sizeof(double);

        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public CudaDeviceVariable<byte> ReductionStorage { get; set; }

        public CudaDeviceVariable<byte> DropoutStorage { get; set; }

        public CudaDeviceVariable<byte> DropoutStateStorage{ get; set; }
        
        public void Dispose()
        {
            Dispose(true);
        }

        public override void Clear()
        {
            Debug.Assert(!this._disposed);

            switch (this.Location)
            {
                case DataLocation.Host:
                    {
                        for (var i = 0; i < this.Shape.TotalLength; i++)
                        {
                            this.HostBuffer[i] = 0.0;
                        }
                    }
                    break;
                case DataLocation.Device:
                    {
                        var res = DriverAPINativeMethods.Memset.cuMemsetD32_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.Size * 2);
                        if (res != CUResult.Success)
                        {
                            throw new CudaException(res);
                        }
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public override void CopyFrom(VolumeStorage<double> source)
        {
            Debug.Assert(!this._disposed);

            var real = source as VolumeStorage;

            if (!ReferenceEquals(this, real))
            {
                if (this.Shape.TotalLength != real.Shape.TotalLength)
                {
                    throw new ArgumentException($"{nameof(real)} has different length!");
                }

                real.CopyToDevice();

                if (this.DeviceBuffer == null)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
                }

                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy(
                    this.DeviceBuffer.DevicePointer,
                    real.DeviceBuffer.DevicePointer,
                    this.Shape.TotalLength * sizeof(double));

                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                this.Location = DataLocation.Device;
            }
            else
            {
                CopyToDevice();
            }
        }

        public void CopyToDevice()
        {
            Debug.Assert(!this._disposed);

            if (this.Location == DataLocation.Host)
            {
                // Device 
                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(
                    this.DeviceBuffer.DevicePointer, this._hostPointer.Start, this.DeviceBuffer.SizeInBytes,
                    this.Context.DefaultStream.Stream);

                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.Location = DataLocation.Device;
            }

            if (_originalStorage != null)
            {
                _originalStorage.Location = this.Location;
            }
        }

        public void CopyToHost()
        {
            Debug.Assert(!this._disposed);

            if (this.Location == DataLocation.Device)
            {
                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                    new IntPtr(this.HostBuffer),
                    this.DeviceBuffer.DevicePointer, this.DeviceBuffer.SizeInBytes, this.Context.DefaultStream.Stream);

                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.Location = DataLocation.Host;
            }

            if (_originalStorage != null)
            {
                _originalStorage.Location = this.Location;
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            this._disposed = true;

            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (this._hostPointer != null && this.HostBuffer != default(double*))
            {
                if (this._originalStorage == null)
                {
                    this._hostPointer.Dispose();
                }
            }

            this.DeviceBuffer?.Dispose();

            this.ConvolutionBackwardFilterStorage?.Dispose();
            this.ConvolutionBackwardStorage?.Dispose();
            this.ConvolutionStorage?.Dispose();
            this.ReductionStorage?.Dispose();
            this.DropoutStorage?.Dispose();
            this.DropoutStateStorage?.Dispose();
        }

        private static void FillWithZeroes(IntPtr memoryStart, long size)
        {
            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                case PlatformID.Win32Windows:
                case PlatformID.WinCE:
                    ZeroMemory(memoryStart, (UIntPtr)size);
                    break;
                default:
                    var buffer = (byte*)memoryStart;
                    for (var i = 0; i < size; i++)
                    {
                        buffer[i] = 0;
                    }
                    break;
            }
        }

        ~VolumeStorage()
        {
            Dispose(false);
        }

        public override double Get(int[] coordinates)
        {
            CopyToHost();
            var length = coordinates.Length;
            return Get(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0);
        }

        public override double Get(int w, int h, int c, int n)
        {
            CopyToHost();
            return this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)];
        }

        public override double Get(int w, int h, int c)
        {
            CopyToHost();
            return
                this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)];
        }

        public override double Get(int w, int h)
        {
            CopyToHost();
            return this.HostBuffer[w + h * this.Shape.GetDimension(0)];
        }

        public override double Get(int i)
        {
            CopyToHost();
            return this.HostBuffer[i];
        }

        private static CudaHostMemoryRegion InitializeSharedMemory(long elementCount)
        {
            var sharedMemory = new CudaHostMemoryRegion(elementCount * sizeof(double));

            // Zero out
            FillWithZeroes(sharedMemory.Start, sharedMemory.ByteCount);
            return sharedMemory;
        }

        public override void Set(int[] coordinates, double value)
        {
            CopyToHost();
            var length = coordinates.Length;
            Set(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0, value);
        }

        public override void Set(int w, int h, int c, int n, double value)
        {
            CopyToHost();
            this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)] = value;
        }

        public override void Set(int w, int h, int c, double value)
        {
            CopyToHost();
            this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)] =
                value;
        }

        public override void Set(int w, int h, double value)
        {
            CopyToHost();
            this.HostBuffer[w + h * this.Shape.GetDimension(0)] = value;
        }

        public override void Set(int i, double value)
        {
            CopyToHost();
            this.HostBuffer[i] = value;
        }

        public override double[] ToArray()
        {
            CopyToHost();
            var array = new double[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int)this.Shape.TotalLength);
            return array;
        }

        [DllImport("Kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
        private static extern void ZeroMemory(IntPtr dest, UIntPtr size);
    }
}