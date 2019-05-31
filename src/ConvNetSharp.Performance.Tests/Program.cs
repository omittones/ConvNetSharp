using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Performance.Tests
{
    public struct Result
    {
        internal long IterationTimeMs;
        internal long TotalTimeMs;
        internal double ForwardTimeMs;
        internal double BackwardTimeMs;
        internal double UpdateWeightsMs;
    }

    public static class Program
    {
        public static void Main(string[] args)
        {
            var gpuVolumeBuilder = new Volume.GPU.Double.VolumeBuilder();
            var cpuVolumeBuilder = new Volume.Double.VolumeBuilder();

            const int nmSets = 400;
            const int nmIterations = 10;
            var inputShape = Shape.From(40, 1, 1);
            var outputShape = Shape.From(1, 1, 4);
            var layerSizes = new[] { 100, 50 };

            int prevBatchSize = 0;
            for (var batchSize = 20; batchSize < nmSets; batchSize = (int)(batchSize * 1.2))
            {
                if (prevBatchSize == batchSize)
                    batchSize += 1;
                prevBatchSize = batchSize;

                Console.WriteLine($"-- {nameof(batchSize)} == {batchSize} ------------------");

                BuilderInstance<double>.Volume = cpuVolumeBuilder;
                var cpuTestNet = CreateNet(inputShape, outputShape, layerSizes);
                var cpuResult = ExecuteNet(cpuTestNet, batchSize, nmSets, nmIterations);

                BuilderInstance<double>.Volume = gpuVolumeBuilder;
                var gpuTestNet = CreateNet(inputShape, outputShape, layerSizes);
                var gpuResult = ExecuteNet(gpuTestNet, batchSize, nmSets, nmIterations);

                DisplayResult(cpuResult, gpuResult);
            }
        }

        private static TestNet CreateNet(Shape input, Shape output, params int[] layerSizes)
        {
            var net = new TestNet();
            net.InputShape = new[] { Shape.From(input) };
            net.OutputShape = Shape.From(output);
            net.AddLayer(new InputLayer(input.Dimensions[0], input.Dimensions[1], input.Dimensions[2]));
            for (var i = 0; i < layerSizes.Length; i++)
            {
                net.AddLayer(new FullyConnLayer(layerSizes[i]));
                net.AddLayer(new ReluLayer());
            }
            net.AddLayer(new FullyConnLayer((int)output.TotalLength));
            net.AddLayer(new SoftmaxLayer((int)output.TotalLength));
            return net;
        }

        private static Result ExecuteNet(
            TestNet net,
            int batchSize,
            int totalSets,
            int iterations)
        {
            var inputs = CreateSampleSets(net, batchSize, totalSets);

            var stopWatch = new Stopwatch();
            stopWatch.Restart();

            var trainer = new SgdTrainer(net);
            trainer.LearningRate = 0.01;
            trainer.Momentum = 0.5;

            for (var i = 0; i < iterations; i++)
            {
                foreach (var set in inputs)
                {
                    trainer.Train(set.Inputs[0], set.Outputs);
                }
            }

            stopWatch.Stop();

            return new Result
            {
                IterationTimeMs = stopWatch.ElapsedMilliseconds / iterations,
                TotalTimeMs = stopWatch.ElapsedMilliseconds,
                ForwardTimeMs = trainer.ForwardTimeMs,
                BackwardTimeMs = trainer.BackwardTimeMs,
                UpdateWeightsMs = trainer.UpdateWeightsTimeMs
            };
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
                        var inputBatch = Shape.From(inputShape.Dimensions[0], inputShape.Dimensions[1], inputShape.Dimensions[2], batchSize);
                        return builder.Random(inputBatch);
                    }).ToArray();

                var outputShape = Shape.From(consumer.OutputShape.Dimensions[0], consumer.OutputShape.Dimensions[1], consumer.OutputShape.Dimensions[2], batchSize);
                var tempBatchOutputs = builder.Random(outputShape);
                var batchOutputs = builder.SameAs(outputShape);
                tempBatchOutputs.Softmax(batchOutputs);

                sets.Add(new Set
                {
                    Inputs = batchInputs,
                    Outputs = batchOutputs
                });
            }

            return sets.ToArray();
        }


        private static void DisplayResult(Result proc, Result gpu)
        {
            Console.WriteLine("                 CPU    |        GPU   ");
            Console.WriteLine("------------------------+--------------");
            Console.WriteLine("iteration: {0,10:0.000}ms | {1,10:0.000}ms", proc.IterationTimeMs, gpu.IterationTimeMs);
            Console.WriteLine("    total: {0,10:0.000}ms | {1,10:0.000}ms", proc.TotalTimeMs, gpu.TotalTimeMs);
            Console.WriteLine("  forward: {0,10:0.000}ms | {1,10:0.000}ms", proc.ForwardTimeMs, gpu.ForwardTimeMs);
            Console.WriteLine(" backward: {0,10:0.000}ms | {1,10:0.000}ms", proc.BackwardTimeMs, gpu.BackwardTimeMs);
            Console.WriteLine("   update: {0,10:0.000}ms | {1,10:0.000}ms", proc.UpdateWeightsMs, gpu.UpdateWeightsMs);
            Console.WriteLine();
        }
    }
}
