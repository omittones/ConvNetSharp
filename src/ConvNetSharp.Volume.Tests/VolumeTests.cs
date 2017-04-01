using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public class SingleVolumeTests : VolumeTests<float>
    {
        protected override float One => 1;

        protected override float OneMinusPDivQ(float p, float q)
        {
            return 1 - p/q;
        }

        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float) i).ToArray();
            return new Single.Volume(converted, shape);
        }
    }

    [TestClass]
    public class DoubleVolumeTests : VolumeTests<double>
    {
        protected override double One => 1;

        protected override double OneMinusPDivQ(double p, double q)
        {
            return 1 - p/q;
        }

        protected override Volume<double> NewVolume(double[] values, Shape shape)
        {
            return new Double.Volume(values, shape);
        }
    }

    public abstract class VolumeTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected abstract T One { get; }
        protected abstract T OneMinusPDivQ(T p, T q);
        protected abstract Volume<T> NewVolume(double[] values, Shape shape);

        [TestMethod]
        public void Add1D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));

            var result = left + right;
            AssertNumber.AreEqual(2.0, result.Get(0));
            AssertNumber.AreEqual(4.0, result.Get(1));
            AssertNumber.AreEqual(6.0, result.Get(2));
        }

        [TestMethod]
        public void Add2D()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            var right = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));

            var result = left + right;
            AssertNumber.AreEqual(2.0, result.Get(0, 0));
            AssertNumber.AreEqual(4.0, result.Get(1, 0));
            AssertNumber.AreEqual(6.0, result.Get(0, 1));
            AssertNumber.AreEqual(8.0, result.Get(1, 1));
        }

        [TestMethod]
        public void AddBroadcast()
        {
            var volume = NewVolume(new[]
            {
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0
            }, new Shape(2, 2, 3));

            var bias = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3));

            var result = volume + bias;
            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0));
            AssertNumber.AreEqual(3.0, result.Get(0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void BiasBackward()
        {
            var outputGradient = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 1));

            var biasGradient = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            AssertNumber.AreEqual(3.0, biasGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(4.0, biasGradient.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(5.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void BiasBackwardBatch()
        {
            var outputGradient = NewVolume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0,
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 2));

            var biasGradient = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            AssertNumber.AreEqual(6.0, biasGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(8.0, biasGradient.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(10.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void Builder()
        {
            var example = NewVolume(new[] {1.0}, new Shape(1));
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, One, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is filled with provided value
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(1.0, volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderArray()
        {
            var array = new[] {1.0, 2.0, 3.0, 4.0, 5.0};
            var volume = NewVolume(array, new Shape(5));

            AssertNumber.AreEqual(5, volume.Shape.GetDimension(0));
            for (var i = 0; i < 5; i++)
            {
                AssertNumber.AreEqual(array[i], volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderEmpty()
        {
            var example = NewVolume(new[] { 1.0 }, new Shape(1));
            var volume = BuilderInstance<T>.Volume.SameAs(example.Storage, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is empty
            Assert.AreEqual(10, volume.Shape.GetDimension(0));

            for (var i = 0; i < 10; i++)
            {
                AssertNumber.AreEqual(0.0, volume.Get(i));
            }
        }

        [TestMethod]
        public void Convolve()
        {
            // 3x3x3x1
            var input = NewVolume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x1
            AssertNumber.AreEqual(1, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void ConvolveBatch()
        {
            // 3x3x3x2
            var input = NewVolume(new double[27 * 2].Populate(1.0), new Shape(3, 3, 3, 2));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x2
            AssertNumber.AreEqual(1, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(24.0, result.Storage.Get(0, 0, 1, 1));
        }

        [TestMethod]
        public void ConvolveGradient()
        {
            // 3x3x3x1
            var input = NewVolume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var outputGradient = NewVolume(new[] { 2.0, 3.0 }, new Shape(1, 1, 2, 1));

            var inputGradient = BuilderInstance<T>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<T>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            AssertNumber.AreEqual(8, inputGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0, inputGradient.Get(2, 2, 2, 0));
            AssertNumber.AreEqual(0, inputGradient.Get(2, 2, 1, 0));
        }

        [TestMethod]
        public void ConvolveGradientBatch()
        {
            // 3x3x3x2
            var input = NewVolume(new double[27 * 2].Populate(1.0), new Shape(3, 3, 3, 2));

            // 2x2x3x2
            var filter = NewVolume(
                new double[12].Populate(1.0).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var outputGradient = NewVolume(new[]
            {
                2.0, 3.0,
                4.0, 5.0
            }, new Shape(1, 1, 2, 2));

            var inputGradient = BuilderInstance<T>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<T>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            // input gradient
            AssertNumber.AreEqual(8.0, inputGradient.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 2, 0));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 1, 0));

            AssertNumber.AreEqual(14.0, inputGradient.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 2, 1));
            AssertNumber.AreEqual(0.0, inputGradient.Get(2, 2, 1, 1));

            // filter gradient
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(1.0, filter.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(2.0, filter.Get(0, 0, 2, 1));
        }

        [TestMethod]
        public void ConvolveForward()
        {
            // 3x3x3x1
            var input = new Single.Volume(new float[27].Populate(1.0f), new Shape(3, 3, 3, 1));

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x1
            AssertNumber.AreEqual(1, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0f, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(24.0f, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void ConvolveForwardBatch()
        {
            // 3x3x3x2
            var input = new Single.Volume(new float[27 * 2].Populate(1.0f), new Shape(3, 3, 3, 2));

            // 2x2x3x2
            var filter = new Single.Volume(
                new float[12].Populate(1.0f).Concat(new float[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2));

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x2
            AssertNumber.AreEqual(1, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(12.0f, result.Storage.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(24.0f, result.Storage.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(12.0f, result.Storage.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(24.0f, result.Storage.Get(0, 0, 1, 1));
        }
        /// <summary>
        ///     Fully connection can be expressed as a convolution with 1x1 filters
        /// </summary>
        [TestMethod]
        public void FullyCon()
        {
            // 1x3x1x1
            var input = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3, 1));

            // 1x1x3x2
            var filter = NewVolume(
                new[] { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 },
                new Shape(1, 1, 3, 2));

            var result = input.Convolve(filter, 0, 1);

            // 1x1x2x1
            AssertNumber.AreEqual(1, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(6.0, result.Storage.Get(0, 0, 0));
            AssertNumber.AreEqual(12.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void Negate()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));

            var result = -volume;
            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(-2.0, result.Get(1));
            AssertNumber.AreEqual(-3.0, result.Get(2));
        }

        [TestMethod]
        public void Pool2D()
        {
            var volume = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0
            }, new Shape(4, 4));

            var result = volume.Pool(2, 2, 0, 2);

            AssertNumber.AreEqual(2, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(1));

            AssertNumber.AreEqual(1.0, result.Get(0, 0));
            AssertNumber.AreEqual(7.0, result.Get(1, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 1));
            AssertNumber.AreEqual(4.0, result.Get(1, 1));
        }

        [TestMethod]
        public void Pool2DBatch()
        {
            var volume = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2));

            var result = volume.Pool(2, 2, 0, 2);

            AssertNumber.AreEqual(2, result.Shape.GetDimension(0));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(1));
            AssertNumber.AreEqual(1, result.Shape.GetDimension(2));
            AssertNumber.AreEqual(2, result.Shape.GetDimension(3));

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(7.0, result.Get(1, 0, 0, 0));
            AssertNumber.AreEqual(2.0, result.Get(0, 1, 0, 0));
            AssertNumber.AreEqual(4.0, result.Get(1, 1, 0, 0));

            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(14.0, result.Get(1, 0, 0, 1));
            AssertNumber.AreEqual(4.0, result.Get(0, 1, 0, 1));
            AssertNumber.AreEqual(8.0, result.Get(1, 1, 0, 1));
        }

        [TestMethod]
        public void Pool2DGradient()
        {
            var inputActivation = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0
            }, new Shape(4, 4));

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(2, 2));

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            AssertNumber.AreEqual(1.0f, result.Get(0, 0));
            AssertNumber.AreEqual(1.0f, result.Get(3, 1));
            AssertNumber.AreEqual(1.0f, result.Get(0, 2));
            AssertNumber.AreEqual(1.0f, result.Get(2, 3));
        }

        [TestMethod]
        public void Pool2DGradientBatch()
        {
            var inputActivation = NewVolume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2));

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = NewVolume(new[]
            {
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            }, new Shape(2, 2, 1, 2));

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            AssertNumber.AreEqual(1.0, result.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(3, 1, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(0, 2, 0, 0));
            AssertNumber.AreEqual(1.0, result.Get(2, 3, 0, 0));

            AssertNumber.AreEqual(2.0, result.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(3, 1, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(0, 2, 0, 1));
            AssertNumber.AreEqual(2.0, result.Get(2, 3, 0, 1));
        }

        [TestMethod]
        public void Relu()
        {
            var volume = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));

            var result = volume.Relu();
            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(3.0, result.Get(2));
            AssertNumber.AreEqual(5.0, result.Get(3));
        }

        [TestMethod]
        public void ReluGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.ReluGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(1.0, result.Get(2));
            AssertNumber.AreEqual(1.0, result.Get(3));
        }

        [TestMethod]
        public void Shape2D()
        {
            var volume = NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(0));
            AssertNumber.AreEqual(2, volume.Shape.GetDimension(1));
        }

        [TestMethod]
        public void Sigmoid()
        {
            var volume = NewVolume(new[] {-1.0, 0.0, 3.0, 5.0}, new Shape(4));

            var result = volume.Sigmoid();
            AssertNumber.AreEqual(1.0/(1.0 + Math.Exp(1.0)), result.Get(0), 0.00001);
            AssertNumber.AreEqual(1.0/(1.0 + Math.Exp(0.0)), result.Get(1), 0.00001);
            AssertNumber.AreEqual(1.0/(1.0 + Math.Exp(-3.0)), result.Get(2), 0.00001);
            AssertNumber.AreEqual(1.0/(1.0 + Math.Exp(-5.0)), result.Get(3), 0.00001);
        }

        [TestMethod]
        public void SigmoidGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.SigmoidGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(0.0, result.Get(0));
            AssertNumber.AreEqual(0.0, result.Get(1));
            AssertNumber.AreEqual(-6.0, result.Get(2));
            AssertNumber.AreEqual(-20.0, result.Get(3));
        }

        [TestMethod]
        public void SoftMax()
        {
            var volume1 = NewVolume(new[] { 0.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            var softmax1 = volume1.SoftMax();
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            var volume2 = NewVolume(new[] { 10000.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1));
            var softmax2 = volume2.SoftMax();
            AssertNumber.AreEqual(0.5, softmax2.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.5, softmax2.Get(0, 0, 3, 0));
        }

        [TestMethod]
        public void SoftMaxBatch()
        {
            var volume1 = NewVolume(new[]
            {
                0.0, 0.0, 0.0, 10000.0,
                0.0, 0.0, 10000.0, 0.0
            }, new Shape(1, 1, -1, 2));
            var softmax1 = volume1.SoftMax();

            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 2, 0));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 0, 1));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 1, 1));
            AssertNumber.AreEqual(1.0, softmax1.Get(0, 0, 2, 1));
            AssertNumber.AreEqual(0.0, softmax1.Get(0, 0, 3, 1));
        }

        [TestMethod]
        public void SoftMaxGradient()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = NewVolume(new[] { 1.0, 0.1, 0.1, 0.1 }, new Shape(1, 1, -1, 1));

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = NewVolume(new[] { 0.0, 1.0, 0.0, 0.0 }, new Shape(1, 1, -1, 1));

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = NewVolume(new double[4], new Shape(1, 1, -1, 1));
            groundTruth.Storage.Map(OneMinusPDivQ, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            AssertNumber.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
        }

        [TestMethod]
        public void SoftMaxGradientBatch()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = NewVolume(new[]
            {
                1.0, 0.1, 0.1, 0.1,
                0.1, 0.1, 1.0, 0.1
            }, new Shape(1, 1, -1, 2));

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = NewVolume(new[]
            {
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            }, new Shape(1, 1, -1, 2));

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = NewVolume(new double[8], new Shape(1, 1, -1, 2));
            groundTruth.Storage.Map(OneMinusPDivQ, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            AssertNumber.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 0, 1), inputGradient.Get(0, 0, 0, 1), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 1, 1), inputGradient.Get(0, 0, 1, 1), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 2, 1), inputGradient.Get(0, 0, 2, 1), 1e-4);
            AssertNumber.AreEqual(result.Get(0, 0, 3, 1), inputGradient.Get(0, 0, 3, 1), 1e-4);
        }

        [TestMethod]
        public void SubstractFrom()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));

            var result = left - right;
            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void DoSubstractFrom()
        {
            var left = NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3));
            var right = NewVolume(new[] { 2.0, 0.0, 1.0 }, new Shape(3));
            var result = BuilderInstance<T>.Volume.SameAs(left.Shape);

            right.DoSubtractFrom(left, result);

            AssertNumber.AreEqual(-1.0, result.Get(0));
            AssertNumber.AreEqual(2.0, result.Get(1));
            AssertNumber.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void Tanh()
        {
            var volume = NewVolume(new[] {-1.0, 0.0, 3.0, 5.0}, new Shape(4));

            var result = volume.Tanh();
            AssertNumber.AreEqual(Math.Tanh(-1.0), result.Get(0), 0.000001);
            AssertNumber.AreEqual(Math.Tanh(0.0), result.Get(1), 0.000001);
            AssertNumber.AreEqual(Math.Tanh(3.0), result.Get(2), 0.000001);
            AssertNumber.AreEqual(Math.Tanh(5.0), result.Get(3), 0.000001);
        }

        [TestMethod]
        public void TanhGradient()
        {
            var inputActivation = NewVolume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4));
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4));

            var result = outputActivation.TanhGradient(inputActivation, outputActivationGradient);

            AssertNumber.AreEqual(1.0, result.Get(0));
            AssertNumber.AreEqual(1.0, result.Get(1));
            AssertNumber.AreEqual(-8.0, result.Get(2));
            AssertNumber.AreEqual(-24.0, result.Get(3));
        }

        [TestMethod]
        public void ToArray()
        {
            var doubles = new[] {1.0, 2.0, 3.0};
            var v = NewVolume(doubles, new Shape(3));

            var array = v.ToArray();

            Assert.AreNotSame(doubles, array);
            foreach (var pair in doubles.Zip(array, (a, b) => new {a, b}))
                AssertNumber.AreEqual(pair.a, pair.b);
        }
    }
}