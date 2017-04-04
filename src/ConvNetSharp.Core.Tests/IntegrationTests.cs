using ConvNetSharp.Core.Layers.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    [TestClass]
    public class IntegrationTests
    {
        [TestMethod]
        public void FeedForwardRelu1Hidden()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer(3, 3, 3));
            net.AddLayer(new FullyConnLayer(10));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new FullyConnLayer(5));
            net.AddLayer(new SoftmaxLayer(5));

            GradientCheckTools.CheckGradientOnNet(net);
        }
    }
}