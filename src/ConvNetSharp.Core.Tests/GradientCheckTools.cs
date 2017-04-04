﻿using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public static class GradientCheckTools
    {
        public struct Sample
        {
            public Volume<double>[] Inputs;
            public Volume<double> Outputs;
        }

        public static void CheckGradientOnNet(Net<double> net, int nmSamples = 100, double epsilon = 1e-10)
        {
            var inputLayers = net.Layers.OfType<InputLayer<double>>().ToArray();
            var lastLayer = net.Layers.OfType<LastLayerBase<double>>().Single();

            var samples = new List<Sample>();
            for (var i = 0; i < nmSamples; i++)
            {
                Sample sample;
                sample.Inputs =
                    inputLayers.Select(
                            l => BuilderInstance.Volume.Random(new Shape(l.InputWidth, l.InputHeight, l.InputDepth)))
                        .ToArray();
                sample.Outputs =
                    BuilderInstance.Volume.Random(new Shape(lastLayer.OutputWidth, lastLayer.OutputHeight,
                        lastLayer.OutputDepth));

                sample.Outputs.MapInplace(v => v*v);
                sample.Outputs = sample.Outputs.SoftMax();
                samples.Add(sample);
            }

            foreach (var sample in samples)
            {
                net.Forward(sample.Inputs, true);
                net.Backward(sample.Outputs);

                var parAndGrads = net.GetParametersAndGradients()
                    .Select(p => new
                    {
                        parameters = p.Volume,
                        calcGradients = p.Gradient.Clone()
                    }).ToArray();

                foreach (var p in parAndGrads)
                {
                    var parameters = p.parameters;
                    var calcGradients = p.calcGradients;

                    for (var x = 0; x < parameters.Shape.GetDimension(0); x++)
                        for (var y = 0; y < parameters.Shape.GetDimension(1); y++)
                            for (var z = 0; z < parameters.Shape.GetDimension(2); z++)
                            {
                                double minusLoss, plusLoss;

                                var calcGradient = calcGradients.Get(x, y, z);

                                var oldValue = parameters.Get(x, y, z);
                                parameters.Set(x, y, z, value: oldValue - epsilon);
                                net.Forward(sample.Inputs, false);
                                lastLayer.Backward(sample.Outputs, out minusLoss);
                                parameters.Set(x, y, z, value: oldValue + epsilon);
                                net.Forward(sample.Inputs, false);
                                lastLayer.Backward(sample.Outputs, out plusLoss);
                                parameters.Set(x, y, z, value: oldValue);

                                var numGradient = (plusLoss - minusLoss)/(2.0*epsilon);

                                Assert.AreEqual(numGradient, calcGradient, calcGradient/1000.0);
                            }
                }
            }
        }

        public static void GradientCheck(LayerBase<double> layer, int inputWidth, int inputHeight, int inputDepth, int bactchSize, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.Random(new Shape(inputWidth, inputHeight, inputDepth, bactchSize), 0.0, Math.Sqrt(2.0 / (inputWidth * inputHeight * inputDepth)));
            var output = layer.DoForward(input, true);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.SameAs(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);
            var computedGradients = layer.InputActivationGradients;

            // Now let's approximate gradient using derivate definition
            for (var d = 0; d < inputDepth; d++)
            {
                for (var y = 0; y < inputHeight; y++)
                {
                    for (var x = 0; x < inputWidth; x++)
                    {
                        var oldValue = input.Get(x, y, d);

                        input.Set(x, y, d, oldValue + epsilon);
                        var output1 = layer.DoForward(input).Clone();
                        input.Set(x, y, d, oldValue - epsilon);
                        var output2 = layer.DoForward(input).Clone();

                        input.Set(x, y, d, oldValue);

                        output1 = output1 - output2;

                        var grad = new double[output.Shape.TotalLength];
                        for (var j = 0; j < output.Shape.TotalLength; j++)
                        {
                            grad[j] = output1.Get(j) / (2.0 * epsilon);
                        }

                        var gradient = grad.Sum(); // approximated gradient
                        var actual = computedGradients.Get(x, y, d);
                        Assert.AreEqual(gradient, actual, 1e-3); // compare layer gradient to the approximated gradient
                    }
                }
            }
        }

        public static void GradienWrtParameterstCheck(int inputWidth, int inputHeight, int inputDepth, int bacthSize, LayerBase<double> layer, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.SameAs(new double[inputWidth * inputHeight * inputDepth * bacthSize].Populate(1.0), new Shape(inputWidth, inputHeight, inputDepth, bacthSize));
            var output = layer.DoForward(input);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.SameAs(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);

            List<ParametersAndGradients<double>> paramsAndGrads = layer.GetParametersAndGradients();

            foreach (var paramAndGrad in paramsAndGrads)
            {
                var vol = paramAndGrad.Volume;
                var gra = paramAndGrad.Gradient;

                // Now let's approximate gradient
                for (var i = 0; i < paramAndGrad.Volume.Shape.TotalLength; i++)
                {
                    input = BuilderInstance<double>.Volume.SameAs(new double[input.Shape.TotalLength].Populate(1.0), input.Shape);

                    var oldValue = vol.Get(i);
                    vol.Set(i, oldValue + epsilon);
                    var output1 = layer.DoForward(input).Clone();
                    vol.Set(i, oldValue - epsilon);
                    var output2 = layer.DoForward(input).Clone();
                    vol.Set(i, oldValue);

                    output1 = output1 - output2;

                    var grad = new double[output.Shape.TotalLength];
                    for (var j = 0; j < output.Shape.TotalLength; j++)
                    {
                        grad[j] = output1.Get(j) / (2.0 * epsilon);
                    }

                    var gradient = grad.Sum(); // approximated gradient
                    Assert.AreEqual(gradient, gra.Get(i), 1e-3); // compare layer gradient to the approximated gradient
                }
            }
        }
    }
}