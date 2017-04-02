using System;
using System.Linq;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class GpuSingleVolumeTests : VolumeTests<float>
    {
        protected override float One => 1.0f;

        protected override float OneMinusPDivQ(float p, float q)
        {
            return 1 - p/q;
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float) i).ToArray();
            return new GPU.Single.Volume(new GPU.Single.VolumeStorage(converted, shape, GpuContext.Default));
        }
    }

    [TestClass]
    public class GpuDoubleVolumeTests : VolumeTests<double>
    {
        protected override double One => 1.0;

        protected override double OneMinusPDivQ(double p, double q)
        {
            return 1 - p/q;
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new GPU.Double.Volume(new GPU.Double.VolumeStorage(values, shape, GpuContext.Default));
        }
    }
}