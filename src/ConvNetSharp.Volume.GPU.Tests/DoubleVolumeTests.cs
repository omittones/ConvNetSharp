﻿using System;
using System.Linq;
using ConvNetSharp.Volume.GPU.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class DoubleVolumeTests
    {
        [TestMethod]
        public void Add1D()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);
            var right = new Double.Volume(new[] { 0.1, 0.2, 0.3 }, new Shape(3), GpuContext.Default);

            var result = left + right;
            Assert.AreEqual(1.1, result.Get(0));
            Assert.AreEqual(2.2, result.Get(1));
            Assert.AreEqual(3.3, result.Get(2));
        }

        [TestMethod]
        public void Add2D()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1), GpuContext.Default);
            var right = new Double.Volume(new[] { 0.1, 0.2, 0.3, 0.4 }, new Shape(2, -1), GpuContext.Default);

            var result = left + right;
            Assert.AreEqual(1.1, result.Get(0, 0));
            Assert.AreEqual(2.2, result.Get(1, 0));
            Assert.AreEqual(3.3, result.Get(0, 1));
            Assert.AreEqual(4.4, result.Get(1, 1));
        }

        [TestMethod]
        public void DoAddToSame()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1), GpuContext.Default);
            var right = new Double.Volume(new[] { 0.1, 0.2, 0.3, 0.4 }, new Shape(2, -1), GpuContext.Default);

            right.DoAdd(left, right);

            Assert.AreEqual(1.1, right.Get(0, 0));
            Assert.AreEqual(2.2, right.Get(1, 0));
            Assert.AreEqual(3.3, right.Get(0, 1));
            Assert.AreEqual(4.4, right.Get(1, 1));
        }

        [TestMethod]
        public void AddBroadcast()
        {
            var volume = new Double.Volume(new[]
            {
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0,
                1.0, 2.0,
                3.0, 4.0
            }, new Shape(2, 2, 3), GpuContext.Default);

            var bias = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3), GpuContext.Default);

            var result = volume + bias;
            Assert.AreEqual(2.0, result.Get(0, 0, 0));
            Assert.AreEqual(3.0, result.Get(0, 0, 1));
            Assert.AreEqual(4.0, result.Get(0, 0, 2));
        }

        [TestMethod]
        public void BiasBackward()
        {
            var outputGradient = new Double.Volume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 1),
                GpuContext.Default);

            var biasGradient = BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            Assert.AreEqual(3.0, biasGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(4.0, biasGradient.Get(0, 0, 1, 0));
            Assert.AreEqual(5.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void BiasBackwardBatch()
        {
            var outputGradient = new Double.Volume(
                new[]
                {
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0,
                    1.0, 2.0,
                    3.0, 1.0,
                    2.0, 3.0
                },
                new Shape(2, 1, 3, 2),
                GpuContext.Default);

            var biasGradient = BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, 3, 1));

            outputGradient.BiasGradient(biasGradient);

            Assert.AreEqual(6.0, biasGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(8.0, biasGradient.Get(0, 0, 1, 0));
            Assert.AreEqual(10.0, biasGradient.Get(0, 0, 2, 0));
        }

        [TestMethod]
        public void Builder()
        {
            var example = new Double.Volume(new[] { 1.0 }, new Shape(1), GpuContext.Default);
            var volume = BuilderInstance<double>.Volume.SameAs(example.Storage, 1.0f, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is filled with provided value
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(1.0f, volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderArray()
        {
            var array = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var volume = BuilderInstance<double>.Volume.SameAs(array, new Shape(5));

            Assert.AreEqual(5, volume.Shape.GetDimension(0));
            for (var i = 0; i < 5; i++)
            {
                Assert.AreEqual(array[i], volume.Get(i));
            }
        }

        [TestMethod]
        public void BuilderEmpty()
        {
            var example = new Double.Volume(new[] { 1.0 }, new Shape(1), GpuContext.Default);
            var volume = BuilderInstance<double>.Volume.SameAs(example.Storage, new Shape(10));

            // SameAs creates an instance that
            // - has the same type of storage as example
            Assert.AreEqual(example.Storage.GetType(), volume.Storage.GetType());
            // - is empty
            Assert.AreEqual(10, volume.Shape.GetDimension(0));
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(0.0, volume.Get(i));
            }
        }

        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            BuilderInstance<double>.Volume = new VolumeBuilder();
        }  
		
        [TestMethod]
        public void Convolve()
        {
            // 3x3x3x1
            var input = new Double.Volume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Double.Volume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2), GpuContext.Default);

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            Assert.AreEqual(12.0, result.Storage.Get(0, 0, 0));
            Assert.AreEqual(24.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void ConvolveBatch()
        {
            // 3x3x3x2
            var input = new Double.Volume(new double[27 * 2].Populate(1.0f), new Shape(3, 3, 3, 2),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Double.Volume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2),
                GpuContext.Default);

            var result = input.Convolve(filter, 0, 2);

            // 1x1x2x2
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            Assert.AreEqual(12.0, result.Storage.Get(0, 0, 0, 0));
            Assert.AreEqual(24.0, result.Storage.Get(0, 0, 1, 0));
            Assert.AreEqual(12.0, result.Storage.Get(0, 0, 0, 1));
            Assert.AreEqual(24.0, result.Storage.Get(0, 0, 1, 1));
        }

        [TestMethod]
        public void ConvolveGradient()
        {
            // 3x3x3x1
            var input = new Double.Volume(new double[27].Populate(1.0), new Shape(3, 3, 3, 1),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Double.Volume(
                new double[12].Populate(1.0f).Concat(new double[12].Populate(2.0)).ToArray(),
                new Shape(2, 2, 3, 2), GpuContext.Default);

            var outputGradient = new Double.Volume(new[] { 2.0, 3.0 }, new Shape(1, 1, 2, 1),
                GpuContext.Default);

            var inputGradient = BuilderInstance<double>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<double>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            Assert.AreEqual(8, inputGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 2, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 1, 0));
        }

        [TestMethod]
        public void ConvolveGradientBatch()
        {
            // 3x3x3x2
            var input = new Double.Volume(new double[27 * 2].Populate(1.0), new Shape(3, 3, 3, 2),
                GpuContext.Default);

            // 2x2x3x2
            var filter = new Double.Volume(
                new double[12].Populate(1.0).Concat(new double[12].Populate(2.0f)).ToArray(),
                new Shape(2, 2, 3, 2),
                GpuContext.Default);

            var outputGradient = new Double.Volume(new[]
                {
                    2.0, 3.0,
                    4.0, 5.0
                }, new Shape(1, 1, 2, 2),
                GpuContext.Default);

            var inputGradient = BuilderInstance<double>.Volume.SameAs(input.Storage, input.Shape);
            var filterGradient = BuilderInstance<double>.Volume.SameAs(filter.Storage, filter.Shape);

            input.ConvolveGradient(filter, outputGradient, inputGradient, filterGradient, 0, 2);

            // input gradient
            Assert.AreEqual(8, inputGradient.Get(0, 0, 0, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 2, 0));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 1, 0));
            Assert.AreEqual(14, inputGradient.Get(0, 0, 0, 1));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 2, 1));
            Assert.AreEqual(0, inputGradient.Get(2, 2, 1, 1));

            // filter gradient
            Assert.AreEqual(1.0, filter.Get(0, 0, 0, 0));
            Assert.AreEqual(1.0, filter.Get(0, 0, 1, 0));
            Assert.AreEqual(1.0, filter.Get(0, 0, 2, 0));
            Assert.AreEqual(2.0, filter.Get(0, 0, 0, 1));
            Assert.AreEqual(2.0, filter.Get(0, 0, 1, 1));
            Assert.AreEqual(2.0, filter.Get(0, 0, 2, 1));
        }

        /// <summary>
        ///     Fully connection can be expressed as a convolution with 1x1 filters
        /// </summary>
        [TestMethod]
        public void FullyCon()
        {
            // 1x3x1x1
            var input = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(1, 1, 3, 1), GpuContext.Default);

            // 1x1x3x2
            var filter = new Double.Volume(
                new[] { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 },
                new Shape(1, 1, 3, 2), GpuContext.Default);

            var result = input.Convolve(filter, 0, 1);

            // 1x1x2x1
            Assert.AreEqual(1, result.Shape.GetDimension(0));
            Assert.AreEqual(1, result.Shape.GetDimension(1));
            Assert.AreEqual(2, result.Shape.GetDimension(2));
            Assert.AreEqual(1, result.Shape.GetDimension(3));

            Assert.AreEqual(6.0, result.Storage.Get(0, 0, 0));
            Assert.AreEqual(12.0, result.Storage.Get(0, 0, 1));
        }

        [TestMethod]
        public void Negate()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);

            var result = -volume;
            Assert.AreEqual(-1.0, result.Get(0));
            Assert.AreEqual(-2.0, result.Get(1));
            Assert.AreEqual(-3.0, result.Get(2));
        }

        [TestMethod]
        public void Pool2D()
        {
            var volume = new Double.Volume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0
            }, new Shape(4, 4), GpuContext.Default);

            var result = volume.Pool(2, 2, 0, 2);

            Assert.AreEqual(2, result.Shape.GetDimension(0));
            Assert.AreEqual(2, result.Shape.GetDimension(0));

            Assert.AreEqual(1.0, result.Get(0, 0));
            Assert.AreEqual(7.0, result.Get(1, 0));
            Assert.AreEqual(2.0, result.Get(0, 1));
            Assert.AreEqual(4.0, result.Get(1, 1));
        }

        [TestMethod]
        public void Pool2DBatch()
        {
            var volume = new Double.Volume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2), GpuContext.Default);

            var result = volume.Pool(2, 2, 0, 2);

            Assert.AreEqual(2, result.Shape.GetDimension(0));
            Assert.AreEqual(2, result.Shape.GetDimension(1));
            Assert.AreEqual(1, result.Shape.GetDimension(2));
            Assert.AreEqual(2, result.Shape.GetDimension(3));

            Assert.AreEqual(1.0, result.Get(0, 0, 0, 0));
            Assert.AreEqual(7.0, result.Get(1, 0, 0, 0));
            Assert.AreEqual(2.0, result.Get(0, 1, 0, 0));
            Assert.AreEqual(4.0, result.Get(1, 1, 0, 0));

            Assert.AreEqual(2.0, result.Get(0, 0, 0, 1));
            Assert.AreEqual(14.0, result.Get(1, 0, 0, 1));
            Assert.AreEqual(4.0, result.Get(0, 1, 0, 1));
            Assert.AreEqual(8.0, result.Get(1, 1, 0, 1));
        }

        [TestMethod]
        public void Pool2DGradient()
        {
            var inputActivation = new Double.Volume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0
            }, new Shape(4, 4), GpuContext.Default);

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = new Double.Volume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(2, 2),
                GpuContext.Default);

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            Assert.AreEqual(1.0f, result.Get(0, 0));
            Assert.AreEqual(1.0f, result.Get(3, 1));
            Assert.AreEqual(1.0f, result.Get(0, 2));
            Assert.AreEqual(1.0f, result.Get(2, 3));
        }

        [TestMethod]
        public void Pool2DGradientBatch()
        {
            var inputActivation = new Double.Volume(new[]
            {
                1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 7.0,
                2.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 4.0, 1.0,

                2.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 2.0, 14.0,
                4.0, 0.0, 2.0, 2.0,
                2.0, 0.0, 8.0, 2.0
            }, new Shape(4, 4, 1, 2), GpuContext.Default);

            var outputActivation = inputActivation.Pool(2, 2, 0, 2);

            var outputActivationGradient = new Double.Volume(new[]
            {
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            }, new Shape(2, 2, 1, 2), GpuContext.Default);

            var result = outputActivation.PoolGradient(inputActivation, outputActivationGradient, 2, 2, 0, 2);

            Assert.AreEqual(1.0, result.Get(0, 0, 0, 0));
            Assert.AreEqual(1.0, result.Get(3, 1, 0, 0));
            Assert.AreEqual(1.0, result.Get(0, 2, 0, 0));
            Assert.AreEqual(1.0, result.Get(2, 3, 0, 0));

            Assert.AreEqual(2.0, result.Get(0, 0, 0, 1));
            Assert.AreEqual(2.0, result.Get(3, 1, 0, 1));
            Assert.AreEqual(2.0, result.Get(0, 2, 0, 1));
            Assert.AreEqual(2.0, result.Get(2, 3, 0, 1));
        }

        [TestMethod]
        public void Relu()
        {
            var volume = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4), GpuContext.Default);

            var result = volume.Relu();
            Assert.AreEqual(0.0, result.Get(0));
            Assert.AreEqual(0.0, result.Get(1));
            Assert.AreEqual(3.0, result.Get(2));
            Assert.AreEqual(5.0, result.Get(3));
        }

        [TestMethod]
        public void ReluGradient()
        {
            var inputActivation = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4),
                GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Double.Volume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.ReluGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(0.0, result.Get(0));
            Assert.AreEqual(0.0, result.Get(1));
            Assert.AreEqual(1.0, result.Get(2));
            Assert.AreEqual(1.0, result.Get(3));
        }

        [TestMethod]
        public void Shape2D()
        {
            var volume = new Double.Volume(new[] { 1.0, 2.0, 3.0, 4.0 }, new Shape(2, -1),
                GpuContext.Default);
            Assert.AreEqual(2, volume.Shape.GetDimension(0));
            Assert.AreEqual(2, volume.Shape.GetDimension(1));
        }

        [TestMethod]
        public void Sigmoid()
        {
            var volume = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4),
                GpuContext.Default);

            var result = volume.Sigmoid();
            Assert.AreEqual(1.0 / (1.0 + Math.Exp(1.0)), result.Get(0));
            Assert.AreEqual(1.0 / (1.0 + Math.Exp(0.0)), result.Get(1));
            Assert.AreEqual(1.0 / (1.0 + Math.Exp(-3.0)), result.Get(2));
            Assert.AreEqual(1.0 / (1.0 + Math.Exp(-5.0)), result.Get(3));
        }

        [TestMethod]
        public void SigmoidGradient()
        {
            var inputActivation = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4),
                GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Double.Volume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.SigmoidGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(0.0, result.Get(0));
            Assert.AreEqual(0.0, result.Get(1));
            Assert.AreEqual(-6.0, result.Get(2));
            Assert.AreEqual(-20.0, result.Get(3));
        }

        [TestMethod]
        public void SoftMax()
        {
            var input1 = new Double.Volume(new[] { 0.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1),
                GpuContext.Default);
            var softmax1 = input1.SoftMax();
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 2, 0));
            Assert.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            var input2 = new Double.Volume(new[] { 10000.0, 0.0, 0.0, 10000.0 }, new Shape(1, 1, -1, 1),
                GpuContext.Default);
            var softmax2 = input2.SoftMax();
            Assert.AreEqual(0.5, softmax2.Get(0, 0, 0, 0));
            Assert.AreEqual(0.5, softmax2.Get(0, 0, 3, 0));
        }

        [TestMethod]
        public void SoftMaxBatch()
        {
            var volume1 = new Double.Volume(new[]
            {
                0.0, 0.0, 0.0, 10000.0,
                0.0, 0.0, 10000.0, 0.0
            }, new Shape(1, 1, -1, 2), GpuContext.Default);
            var softmax1 = volume1.SoftMax();

            Assert.AreEqual(0.0, softmax1.Get(0, 0, 0, 0));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 1, 0));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 2, 0));
            Assert.AreEqual(1.0, softmax1.Get(0, 0, 3, 0));

            Assert.AreEqual(0.0, softmax1.Get(0, 0, 0, 1));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 1, 1));
            Assert.AreEqual(1.0, softmax1.Get(0, 0, 2, 1));
            Assert.AreEqual(0.0, softmax1.Get(0, 0, 3, 1));
        }

        [TestMethod]
        public void SoftMaxGradient()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = new Double.Volume(new[] { 1.0, 0.1, 0.1, 0.1 }, new Shape(1, 1, -1, 1), GpuContext.Default);

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = new Double.Volume(new[] { 0.0, 1.0, 0.0, 0.0 }, new Shape(1, 1, -1, 1),
                GpuContext.Default);

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = new Double.Volume(new double[4], new Shape(1, 1, -1, 1), GpuContext.Default);
            groundTruth.Storage.Map((p, q) => 1 - p / q, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            Assert.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
        }

        [TestMethod]
        public void SoftMaxGradientBatch()
        {
            // input = [1,  0.1, 0.1, 0.1]
            var input = new Double.Volume(new[]
            {
                1.0, 0.1, 0.1, 0.1,
                0.1, 0.1, 1.0, 0.1
            }, new Shape(1, 1, -1, 2), GpuContext.Default);

            // output  = softmax(input)
            var output = input.SoftMax();

            // groundTruth = [0, 1, 0 , 0]
            var groundTruth = new Double.Volume(new[]
            {
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            }, new Shape(1, 1, -1, 2), GpuContext.Default);

            // output gradient = 1 - groundTruth ./ output
            var outputGradient = new Double.Volume(new double[8], new Shape(1, 1, -1, 2), GpuContext.Default);
            groundTruth.Storage.Map((p, q) => 1 - p / q, output.Storage, outputGradient.Storage);

            // inputGradient = softmax_gradient(output, outputGradient)
            var inputGradient = output.SoftMaxGradient(outputGradient);

            // theorical result = output-groundTruth
            var result = output - groundTruth;

            Assert.AreEqual(result.Get(0, 0, 0, 0), inputGradient.Get(0, 0, 0, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 0), inputGradient.Get(0, 0, 1, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 0), inputGradient.Get(0, 0, 2, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 0), inputGradient.Get(0, 0, 3, 0), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 0, 1), inputGradient.Get(0, 0, 0, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 1, 1), inputGradient.Get(0, 0, 1, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 2, 1), inputGradient.Get(0, 0, 2, 1), 1e-4);
            Assert.AreEqual(result.Get(0, 0, 3, 1), inputGradient.Get(0, 0, 3, 1), 1e-4);
        }

        [TestMethod]
        public void SubstractFrom()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3),
                GpuContext.Default);
            var right = new Double.Volume(new[] { 2.0, 0.0, 1.0 }, new Shape(3),
                GpuContext.Default);

            var result = left - right;
            Assert.AreEqual(-1.0, result.Get(0));
            Assert.AreEqual(2.0, result.Get(1));
            Assert.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void DoSubstractFrom()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);
            var right = new Double.Volume(new[] { 2.0, 0.0, 1.0 }, new Shape(3), GpuContext.Default);
            var result = BuilderInstance<double>.Volume.SameAs(left.Shape);

            right.DoSubtractFrom(left, result);

            Assert.AreEqual(-1.0, result.Get(0));
            Assert.AreEqual(2.0, result.Get(1));
            Assert.AreEqual(2.0, result.Get(2));
        }

        [TestMethod]
        public void DoSubstractFromInPlace()
        {
            var left = new Double.Volume(new[] { 1.0, 2.0, 3.0 }, new Shape(3), GpuContext.Default);
            var right = new Double.Volume(new[] { 2.0, 0.0, 1.0 }, new Shape(3), GpuContext.Default);

            right.DoSubtractFrom(left, left);

            Assert.AreEqual(-1.0, left.Get(0));
            Assert.AreEqual(2.0, left.Get(1));
            Assert.AreEqual(2.0, left.Get(2));
        }

        [TestMethod]
        public void Tanh()
        {
            var volume = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4),
                GpuContext.Default);

            var result = volume.Tanh();
            Assert.AreEqual(Math.Tanh(-1.0), result.Get(0));
            Assert.AreEqual(Math.Tanh(0.0), result.Get(1));
            Assert.AreEqual(Math.Tanh(3.0), result.Get(2));
            Assert.AreEqual(Math.Tanh(5.0), result.Get(3));
        }

        [TestMethod]
        public void TanhGradient()
        {
            var inputActivation = new Double.Volume(new[] { -1.0, 0.0, 3.0, 5.0 }, new Shape(4),
                GpuContext.Default);
            var outputActivation = inputActivation.Relu();
            var outputActivationGradient = new Double.Volume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4),
                GpuContext.Default);

            var result = outputActivation.TanhGradient(inputActivation, outputActivationGradient);

            Assert.AreEqual(1.0, result.Get(0));
            Assert.AreEqual(1.0, result.Get(1));
            Assert.AreEqual(-8.0, result.Get(2));
            Assert.AreEqual(-24.0, result.Get(3));
        }

        [TestMethod]
        public void Multiply()
        {
            var matrix = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var a = new Double.Volume(matrix, Shape.From(2, 2, 2), GpuContext.Default);
            var b = a.Clone();

            const double eps = 0.00001;

            var result = b.Multiply(0.1);
            Assert.AreNotSame(b, result);
            Assert.AreNotSame(b.Storage, result.Storage);
            for (var i = 0; i < matrix.Length; i++)
            {
                Assert.AreEqual(matrix[i], a.Get(i), eps);
                Assert.AreEqual(matrix[i], b.Get(i), eps);
                Assert.AreEqual(matrix[i] * 0.1, result.Get(i), eps);
            }

            b = result;
            result = a.Clone();
            a.DoMultiply(b, result);
            for (var i = 0; i < matrix.Length; i++)
            {
                Assert.AreEqual(matrix[i], a.Get(i), eps);
                Assert.AreEqual(matrix[i] * 0.1, b.Get(i), eps);
                Assert.AreEqual(matrix[i] * matrix[i] * 0.1, result.Get(i), eps);
            }
        }

        [TestMethod]
        public void ToArray()
        {
            var doubles = new[] { 1.0, 2.0, 3.0 };
            var v = new Double.Volume(doubles, new Shape(3), GpuContext.Default);

            var array = v.ToArray();

            Assert.IsTrue(doubles.SequenceEqual(array));
        }
    }
}