﻿using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Double
{
    public unsafe class VolumeStorage : VolumeStorage<double>, IDisposable
    {
        private readonly IntPtr _hostPointer;
        private bool _allocatedOnDevice;

        public VolumeStorage(Shape shape, GpuContext context, long length = -1) : base(shape)
        {
            this.Context = context;

            // Take care of unkown dimension
            if (length != -1)
            {
                this.Shape.GuessUnkownDimension(length);
            }

            // Host 
            this._hostPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this._hostPointer, this.Shape.TotalLength * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.HostBuffer = (double*) this._hostPointer;
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
        }

        public VolumeStorage(VolumeStorage storage, Shape shape)
            : this(shape, storage.Context, storage.Shape.TotalLength)
        {
            storage.CopyToHost();

            // Fill host buffer
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                this.HostBuffer[i] = storage.HostBuffer[i];
            }
        }

        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public bool CopiedToDevice { get; set; }

        public bool CopiedToHost { get; set; }

        public double* HostBuffer { get; private set; }

        public CudaDeviceVariable<double> DeviceBuffer { get; private set; }

        public GpuContext Context { get; }

        public void Dispose() => Dispose(true);

        public override void Clear()
        {
            CopyToDevice();

            DriverAPINativeMethods.Memset.cuMemsetD32_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.SizeInBytes);

            this.CopiedToDevice = true;
            this.CopiedToHost = false;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (this.HostBuffer != default(double*))
            {
                var tmp = new IntPtr(this.HostBuffer);
                this.HostBuffer = default(double*);
                try
                {
                    DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }
            }

            this.DeviceBuffer?.Dispose();
            this.ConvolutionBackwardFilterStorage?.Dispose();
            this.ConvolutionBackwardStorage?.Dispose();
            this.ConvolutionStorage?.Dispose();
        }

        ~VolumeStorage()
        {
            Dispose(false);
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

        public override void Set(int w, int h, int c, int n, double value)
        {
            CopyToHost();
            this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)] = value;
            this.CopiedToDevice = false;
        }

        public override void Set(int w, int h, int c, double value)
        {
            CopyToHost();
            this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)] =
                value;
            this.CopiedToDevice = false;
        }

        public override void Set(int w, int h, double value)
        {
            CopyToHost();
            this.HostBuffer[w + h * this.Shape.GetDimension(0)] = value;
            this.CopiedToDevice = false;
        }

        public override void Set(int i, double value)
        {
            CopyToHost();
            this.HostBuffer[i] = value;
            this.CopiedToDevice = false;
        }

        public override double[] ToArray()
        {
            CopyToHost();

            var array = new double[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int) this.Shape.TotalLength);
            return array;
        }

        public void CopyToDevice()
        {
            if (!this.CopiedToDevice)
            {
                // Device 
                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(
                    this.DeviceBuffer.DevicePointer,
                    this._hostPointer,
                    this.DeviceBuffer.SizeInBytes);

                if (res != CUResult.Success)
                    throw new CudaException(res);
            }

            this.CopiedToDevice = true;
            this.CopiedToHost = false;
        }

        public void CopyToHost()
        {
            if (this.CopiedToDevice && !this.CopiedToHost)
            {
                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(
                    new IntPtr(this.HostBuffer),
                    this.DeviceBuffer.DevicePointer,
                    this.DeviceBuffer.SizeInBytes);

                if (res != CUResult.Success)
                    throw new CudaException(res);

                this.CopiedToHost = true;
            }
        }

        public void CopyFrom(VolumeStorage source)
        {
            if (!object.ReferenceEquals(this, source))
            {
                if (this.Shape.TotalLength != source.Shape.TotalLength)
                    throw new ArgumentException($"{nameof(source)} has different length!");

                source.CopyToDevice();

                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy(
                    this.DeviceBuffer.DevicePointer,
                    source.DeviceBuffer.DevicePointer,
                    this.Shape.TotalLength*sizeof(double));

                if (res != CUResult.Success)
                    throw new CudaException(res);

                this.CopiedToDevice = true;
                this.CopiedToHost = false;
            }
            else
            {
                this.CopyToDevice();
            }
        }
    }
}