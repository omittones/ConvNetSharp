using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Training;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.Core.Tests
{
    [TestClass]
    public class TrainerTests
    {
        [TestMethod]
        public void ChangingInputDoesNotChangeState()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer<double>(1, 1, 1));
            net.AddLayer(new FullyConnLayer<double>(3));
            net.AddLayer(new RegressionLayer<double>());

            var dqn = new DQNTrainer(net, 3);
            dqn.ReplayMemorySize = 1000;
            dqn.ReplaySkipCount = 0;

            var inputs = new double[] { 1, 2, 3 };
            var first = dqn.Act(inputs);

            inputs[0] = 0;
            inputs[1] = 0;
            inputs[2] = 0;

            var second = dqn.Act(inputs);

            Assert.AreEqual(1, first.State[0]);
            Assert.AreEqual(2, first.State[1]);
            Assert.AreEqual(3, first.State[2]);

            Assert.AreEqual(0, second.State[0]);
            Assert.AreEqual(0, second.State[1]);
            Assert.AreEqual(0, second.State[2]);
        }

        [TestMethod]
        public void GradientsAccumulate()
        {
            var net = new Net<double>();
            net.AddLayer(new InputLayer<double>(2, 2, 2));
            net.AddLayer(new FullyConnLayer<double>(10));
            net.AddLayer(new LeakyReluLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(5));
            net.AddLayer(new LeakyReluLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            net.AddLayer(new SoftmaxLayer<double>());

            net.Forward(new[] { 1, 2, 3, 4, 5, 6, 7, 8.0 }, true);
            net.Backward(new[] { 0, 1.0 });

            var firstBatch =
                net.GetParametersAndGradients()
                .SelectMany(e => e.Gradient.ToArray())
                .ToArray();

            foreach (var grad in firstBatch)
                Assert.AreNotEqual(0, grad, double.Epsilon);

            net.Forward(new[] { 1, 2, 3, 4, 5, 6, 7, 8.0 }, true);
            net.Backward(new[] { 0, 1.0 });

            var secondBatch =
             net.GetParametersAndGradients()
             .SelectMany(e => e.Gradient.ToArray())
             .ToArray();

            var factors = firstBatch
                .Zip(secondBatch, (f, s) => s / f)
                .ToArray();

            foreach (var fact in factors)
                Assert.AreEqual(2, fact, double.Epsilon);
        }

        [TestMethod]
        public void GradientsScale()
        {
            var net = new Net<double>();
            var lastLayer = new SoftmaxLayer<double>();
            net.AddLayer(new InputLayer<double>(2, 2, 2));
            net.AddLayer(new FullyConnLayer<double>(10));
            net.AddLayer(new LeakyReluLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(5));
            net.AddLayer(new LeakyReluLayer<double>());
            net.AddLayer(new FullyConnLayer<double>(2));
            net.AddLayer(lastLayer);

            net.Forward(new[] { 1, 2, 3, 4, 5, 6, 7, 8.0 }, true);
            net.Backward(new[] { 0.0, 1.0 });

            var firstBatch =
                net.GetParametersAndGradients()
                .SelectMany(e => e.Gradient.ToArray())
                .ToArray();

            foreach (var grad in firstBatch)
                Assert.AreNotEqual(0, grad, double.Epsilon);

            net.GetParametersAndGradients()
                .ForEach(g => g.Gradient.Clear());

            lastLayer.BatchRewards = new[] { 1000.0 };
            net.Forward(new[] { 1, 2, 3, 4, 5, 6, 7, 8.0 }, true);
            net.Backward(new[] { 0.0, 1.0 });

            var secondBatch =
             net.GetParametersAndGradients()
             .SelectMany(e => e.Gradient.ToArray())
             .ToArray();

            var factors = firstBatch
                .Zip(secondBatch, (f, s) => s / f)
                .ToArray();

            foreach (var fact in factors)
                Assert.AreEqual(1000.0, fact, 0.000001);
        }
    }
}