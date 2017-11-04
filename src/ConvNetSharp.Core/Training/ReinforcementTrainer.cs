﻿using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;

namespace ConvNetSharp.Core.Training
{
    public class ReinforcementTrainer<T> : SgdTrainer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        public T Baseline { get; private set; }
        public T RewardDiscountGamma { get; set; }
        public T EstimatedRewards => Ops<T>.Negate(this.Loss);

        private readonly IReinforcementLayer<T> finalLayer;
        private readonly Random rnd;

        public ReinforcementTrainer(
            Net<T> net,
            Random rnd) : base(net)
        {
            this.finalLayer = net.Layers
                .OfType<IReinforcementLayer<T>>()
                .FirstOrDefault();

            this.rnd = rnd;
            this.Baseline = Ops<T>.Zero;
            this.RewardDiscountGamma = Ops<T>.Zero;
        }

        public int[] Act(Volume<T> inputs)
        {
            var output = this.Forward(inputs);

            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == this.BatchSize);

            var classCount = output.Depth;
            var values = new T[classCount];
            var prediction = new int[this.BatchSize];
            for (var n = 0; n < this.BatchSize; n++)
            {

                int max = 0;
                for (var j = 0; j < classCount; j++)
                    if (Ops<T>.GreaterThan(output.Get(0, 0, j, n), output.Get(0, 0, max, n)))
                        max = j;
                prediction[n] = max;
            }

            return prediction;
        }
        
        public override void Train(Volume<T> x, Volume<T> y)
        {
            throw new NotSupportedException("Use Reinforce method instead!");
        }

        public void Reinforce(Volume<T> pathInputs, int[][] pathActions, T[] pathReturns)
        {
            this.Forward(pathInputs);

            this.finalLayer.SetReturns(pathActions, pathReturns, this.Baseline, this.RewardDiscountGamma);

            //refit baseline to minimize Sum[(R - b)^2]
            Baseline = Ops<T>.Zero;
            for (var i = 0; i < pathReturns.Length; i++)
                Baseline = Ops<T>.Add(pathReturns[i], Baseline);
            Baseline = Ops<T>.Divide(Baseline, Ops<T>.Cast(pathReturns.Length));
 
            //var mean = averages.Average();
            //averages = averages.Select(a => a - mean).ToArray();
            //var stdev = averages.Select(a => a * a).Average();
            //averages = averages.Select(a => a / stdev).ToArray();

            this.Backward(pathInputs);

            var chrono = Stopwatch.StartNew();

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / pathInputs.BatchSize;
        }
    }
}
