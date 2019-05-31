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
    }
}