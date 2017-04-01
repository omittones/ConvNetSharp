using System;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    [TestClass]
    public class SingleSoftMaxLayerTests : SoftMaxLayerTests<float>
    {
    }

    [TestClass]
    public class DoubleSoftMaxLayerTests : SoftMaxLayerTests<double>
    {
    }

    public class SoftMaxLayerTests<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        private SoftmaxLayer<T> layer;
        private Volume<T> input;

        [TestInitialize]
        public void Initialize()
        {
            this.layer = new SoftmaxLayer<T>(4);
            this.layer.Init(1, 1, 4);
            this.input = BuilderInstance<T>.Volume.SameAs(new[]
            {
                0.1, 0.1, 0.1, 0.1,
                1000, 2000, 3000, 4000,
                0, 0, 0, 0
            }.To<T>(), new Shape(1, 1, 4, 3));
        }

        [TestMethod]
        public void OutputIsNormalized()
        {
            var output = this.layer.DoForward(input, true);
            Assert.AreEqual(1, output.Shape.GetDimension(0));
            Assert.AreEqual(1, output.Shape.GetDimension(1));
            Assert.AreEqual(4, output.Shape.GetDimension(2));
            Assert.AreEqual(3, output.Shape.GetDimension(3));

            var values = output.ToArray();
            AssertNumber.AreEqual(0.25, values[0]);
            AssertNumber.AreEqual(0.25, values[1]);
            AssertNumber.AreEqual(0.25, values[2]);
            AssertNumber.AreEqual(0.25, values[3]);

            AssertNumber.AreEqual(0, values[4]);
            AssertNumber.AreEqual(0, values[5]);
            AssertNumber.AreEqual(0, values[6]);
            AssertNumber.AreEqual(1, values[7]);

            AssertNumber.AreEqual(0.25, values[8]);
            AssertNumber.AreEqual(0.25, values[9]);
            AssertNumber.AreEqual(0.25, values[10]);
            AssertNumber.AreEqual(0.25, values[11]);
        }

        [TestMethod]
        public void StorageIsReusedIfPossible()
        {
            var output1 = this.layer.DoForward(input, true);
            var output2 = this.layer.DoForward(input, true);
            Assert.AreSame(output1, output2, "Storage is reused if possible.");
        }
    }
}