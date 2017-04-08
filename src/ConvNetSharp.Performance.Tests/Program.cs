﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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

    public class Program
    {
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

                sets.Add(new Tests.Set
                {
                    Inputs = batchInputs,
                    Outputs = batchOutputs
                });
            }

            return sets.ToArray();
        }

        private static TestNet Create(int layerSize, int nmLayers, int inputWHD)
        {
            var net = new TestNet();
            net.InputShape = new[] {Shape.From(inputWHD, inputWHD, inputWHD)};
            net.OutputShape = Shape.From(1, 1, layerSize);
            net.AddLayer(new InputLayer(inputWHD, inputWHD, inputWHD));
            for (var i = 0; i < nmLayers; i++)
            {
                net.AddLayer(new FullyConnLayer(layerSize));
                net.AddLayer(new SigmoidLayer());
            }
            net.AddLayer(new FullyConnLayer(layerSize));
            net.AddLayer(new SoftmaxLayer(layerSize));
            return net;
        }

        public static void Main(string[] args)
        {
            var gpuVolumeBuilder = new Volume.GPU.Double.VolumeBuilder();
            var cpuVolumeBuilder = new Volume.Double.VolumeBuilder();

            BuilderInstance<double>.Volume = cpuVolumeBuilder;
            var testNet = Create(60, 10, 10);
            ExecuteNeuralNet("CPU", testNet, 50, 1000, 1);

            BuilderInstance<double>.Volume = gpuVolumeBuilder;
            testNet = Create(60, 10, 10);
            ExecuteNeuralNet("GPU", testNet, 50, 1000, 1);
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
            trainer.Momentum = 0;
            trainer.BatchSize = batchSize;

            for (var i = 0; i < iterations; i++)
            {
                foreach (var set in inputs)
                {
                    trainer.Train(set.Inputs[0], set.Outputs);
                }
            }

            stopWatch.Stop();
            Console.WriteLine(stopWatch.ElapsedMilliseconds);
        }

        //private static void AddOneToBlock(VolumeBuilder<double> builder)
        //{
        //    var volume = builder.SameAs(new[] {1.0, 2, 3, 4}, Shape.From(2, 2, 1, 1));
        //    var one = builder.SameAs(new[] {1.0}, Shape.From(1, 1, 1, 1));
        //    volume.DoAdd(one, volume);
        //    volume.Get(0);
        //}
    }
}
