using ConvNetSharp.Core.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public class DoubleIntegrationTests : IntegrationTests<double>
    {
        public override double Epsilon => 1e-10;

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
    public class SingleIntegrationTests : IntegrationTests<float>
    {
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