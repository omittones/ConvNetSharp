using ConvNetSharp.Core.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU
{
    [TestClass]
    public class GpuDoubleIntegrationTests : IntegrationTests<double>
    {
        public GpuDoubleIntegrationTests()
        {
            BuilderInstance<double>.Volume = new GPU.Double.VolumeBuilder();
        }

        public override double Epsilon => 1e-7;

        [TestMethod]
        public override void Minimal()
        {
            base.Minimal();
        }

        [TestMethod]
        public override void FeedForwardRelu1Hidden()
        {
            base.FeedForwardRelu1Hidden();
        }
    }

    [TestClass]
    public class GpuSingleIntegrationTests : IntegrationTests<float>
    {
        public GpuSingleIntegrationTests()
        {
            BuilderInstance<float>.Volume = new GPU.Single.VolumeBuilder();
        }

        public override double Epsilon => 1e-4;

        [TestMethod]
        public override void Minimal()
        {
            base.Minimal();
        }

        [TestMethod]
        public override void FeedForwardRelu1Hidden()
        {
            base.FeedForwardRelu1Hidden();
        }
    }
}