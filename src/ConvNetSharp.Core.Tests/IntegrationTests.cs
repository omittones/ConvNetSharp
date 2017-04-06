using System;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public class IntegrationTests<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        public virtual void FeedForwardRelu1Hidden()
        {
            var net = new Net<T>();
            net.AddLayer(new InputLayer<T>(3, 3, 3));
            net.AddLayer(new FullyConnLayer<T>(10));
            net.AddLayer(new ReluLayer<T>());
            net.AddLayer(new FullyConnLayer<T>(5));
            net.AddLayer(new SoftmaxLayer<T>(5));

            GradientCheckTools.CheckGradientOnNet(net);
        }
    }

    [TestClass]
    public class DoubleIntegrationTests : IntegrationTests<double>
    {
        [TestMethod]
        public override void FeedForwardRelu1Hidden()
        {
            base.FeedForwardRelu1Hidden();
        }
    }

    [TestClass]
    public class SingleIntegrationTests : IntegrationTests<double>
    {
        [TestMethod]
        public override void FeedForwardRelu1Hidden()
        {
            base.FeedForwardRelu1Hidden();
        }
    }
}