using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Performance.Tests
{
    public class TestNet : Net<double>
    {
        public Shape[] InputShape { get; set; }
        public Shape OutputShape { get; set; }
    }

    public class Set
    {
        public Volume<double>[] Inputs { get; set; }
        public Volume<double> Outputs { get; set; }
    }

    public static class Program
    {
        public static void Main(string[] args)
        {
            var gpuVolumeBuilder = new Volume.GPU.Double.VolumeBuilder();
            var cpuVolumeBuilder = new Volume.Double.VolumeBuilder();

            const int nmSets = 2000;
            const int nmIterations = 1;
            var input = Shape.From(590, 1, 1);
            var output = 5;

            for (var batchSize = 130; batchSize < 170; batchSize += 5)
            {
                Console.WriteLine($"-- {nameof(batchSize)} == {batchSize} ------------------");

                BuilderInstance<double>.Volume = cpuVolumeBuilder;
                var testNet = Create(input, output, 30, 30, 30);
                ExecuteNeuralNet("CPU", testNet, batchSize, nmSets, nmIterations);

                BuilderInstance<double>.Volume = gpuVolumeBuilder;
                testNet = Create(input, output, 30, 30, 30);
                ExecuteNeuralNet("GPU", testNet, batchSize, nmSets, nmIterations);

                Console.WriteLine();
            }
        }

        private static TestNet Create(Shape input, int output, params int[] layerSizes)
        {
            var net = new TestNet();
            net.InputShape = new[] { Shape.From(input) };
            net.OutputShape = Shape.From(1, 1, output);
            net.AddLayer(new InputLayer(input.GetDimension(0), input.GetDimension(1), input.GetDimension(2)));
            for (var i = 0; i < layerSizes.Length; i++)
            {
                net.AddLayer(new FullyConnLayer(layerSizes[i]));
                net.AddLayer(new ReluLayer());
            }
            net.AddLayer(new FullyConnLayer(output));
            net.AddLayer(new SoftmaxLayer(output));
            return net;
        }

        public static Set[] CreateSampleSets(
            TestNet consumer,
            int batchSize,
            int totalSets)
        {
            var sets = new List<Set>();

            var builder = BuilderInstance<double>.Volume;

            for (var s = 0; s < totalSets; s += batchSize)
            {
                var batchInputs = consumer
                    .InputShape
                    .Select(inputShape =>
                    {
                        var inputBatch = Shape.From(inputShape, batchSize);
                        return builder.Random(inputBatch);
                    }).ToArray();

                var outputShape = Shape.From(consumer.OutputShape, batchSize);
                var tempBatchOutputs = builder.Random(outputShape);
                var batchOutputs = builder.SameAs(outputShape);
                tempBatchOutputs.DoSoftMax(batchOutputs);

                sets.Add(new Set
                {
                    Inputs = batchInputs,
                    Outputs = batchOutputs
                });
            }

            return sets.ToArray();
        }

        private static void ExecuteNeuralNet(
            string name,
            TestNet net,
            int batchSize,
            int totalSets,
            int iterations)
        {
            var inputs = CreateSampleSets(net, batchSize, totalSets);

            var stopWatch = new Stopwatch();
            Console.WriteLine($"- {name} ------");
            stopWatch.Restart();

            var trainer = new SgdTrainer(net);
            trainer.LearningRate = 0.01;
            trainer.Momentum = 0.5;
            trainer.L1Decay = 0.01;
            trainer.L2Decay = 0.01;
            trainer.BatchSize = batchSize;

            for (var i = 0; i < iterations; i++)
            {
                foreach (var set in inputs)
                {
                    trainer.Train(set.Inputs[0], set.Outputs);
                }
            }

            stopWatch.Stop();

            Console.WriteLine("    total: {0:0.000}ms", stopWatch.ElapsedMilliseconds);
            Console.WriteLine("  forward: {0:0.000}ms", trainer.ForwardTimeMs);
            Console.WriteLine(" backward: {0:0.000}ms", trainer.BackwardTimeMs);
            Console.WriteLine("   update: {0:0.000}ms", trainer.UpdateWeightsTimeMs);
        }
    }
}
