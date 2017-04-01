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
                0.1f, 0.1f, 0.1f, 0.1f,
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
            Assert<T>.AreEqual(0.25f, values[0]);
            Assert<T>.AreEqual(0.25f, values[1]);
            Assert<T>.AreEqual(0.25f, values[2]);
            Assert<T>.AreEqual(0.25f, values[3]);

            Assert<T>.AreEqual(0, values[4]);
            Assert<T>.AreEqual(0, values[5]);
            Assert<T>.AreEqual(0, values[6]);
            Assert<T>.AreEqual(1, values[7]);

            Assert<T>.AreEqual(0.25f, values[8]);
            Assert<T>.AreEqual(0.25f, values[9]);
            Assert<T>.AreEqual(0.25f, values[10]);
            Assert<T>.AreEqual(0.25f, values[11]);
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