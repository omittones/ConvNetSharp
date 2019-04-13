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
            var input = Shape.From(40, 1, 1);
            var output = 4;

            int prevBatchSize = 0;
            for (var batchSize = 20; batchSize < nmSets; batchSize = (int)(batchSize * 1.2))
            {
                if (prevBatchSize == batchSize)
                    batchSize += 1;
                prevBatchSize = batchSize;

                Console.WriteLine($"-- {nameof(batchSize)} == {batchSize} ------------------");

                BuilderInstance<double>.Volume = cpuVolumeBuilder;
                var cpuTestNet = CreateNet(input, output, 50, 30);
                var cpuResult = ExecuteNet(cpuTestNet, batchSize, nmSets, nmIterations);

                BuilderInstance<double>.Volume = gpuVolumeBuilder;
                var gpuTestNet = CreateNet(input, output, 50, 30);
                var gpuResult = ExecuteNet(gpuTestNet, batchSize, nmSets, nmIterations);

                DisplayResult(cpuResult, gpuResult);
            }
        }

        private static TestNet CreateNet(Shape input, int output, params int[] layerSizes)
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
            net.AddLayer(new SoftmaxLayer());
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
            trainer.L1Decay = 0.01;
            trainer.L2Decay = 0.01;

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
                        var inputBatch = Shape.From(inputShape, batchSize);
                        return builder.Random(inputBatch);
                    }).ToArray();

                var outputShape = Shape.From(consumer.OutputShape, batchSize);
                var tempBatchOutputs = builder.Random(outputShape);
                var batchOutputs = builder.SameAs(outputShape);
                tempBatchOutputs.DoSoftmax(batchOutputs);

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
