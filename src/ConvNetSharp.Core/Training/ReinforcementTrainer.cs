using ConvNetSharp.Volume; 
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Training
{
    public class ActionInput
    {
        public int Action;
        public Volume<double> Inputs;

        public override string ToString()
        {
            return "action: " + Action.ToString();
        }
    }

    public class ActionInputReward
    {
        public int Action;
        public Volume<double> Inputs;
        public double Reward;

        public override string ToString()
        {
            return $"{Action} -> {Reward:0.000} reward";
        }
    }

    public class Path : List<ActionInputReward>
    {
        internal bool Used;

        public void SetReward(double reward)
        {
            foreach (var action in this)
                action.Reward = reward;
        }

        public void Add(ActionInput step)
        {
            this.Add(new ActionInputReward
            {
                Action = step.Action,
                Inputs = step.Inputs,
                Reward = 0
            });
        }

        public override string ToString()
        {
            var avg = this.Average(a => a.Reward);
            var actions = string.Join(", ", this.Select(e => e.Action).ToArray());
            return $"[{actions}] -> {avg:0.000} reward";
        }
    }
    
    public class ReinforcementTrainer : SgdTrainer<double>
    {
        public double EstimatedRewards => Ops<double>.Negate(this.Loss);

        private readonly SoftmaxLayer<double> finalLayer;
        private readonly InputLayer<double> inputLayer;
        private Volume<double> input;
        private Volume<double> output;

        public ReinforcementTrainer(Net<double> net) : base(net)
        {
            this.inputLayer = net.Layers
                .OfType<InputLayer<double>>()
                .Single();

            this.finalLayer = net.Layers
                .OfType<SoftmaxLayer<double>>()
                .Last();
        }

        public ActionInput Act(Volume<double> inputs)
        {
            if (inputs.BatchSize != 1)
                throw new NotSupportedException("Not supported!");

            var output = this.Net.Forward(inputs, false);
            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == 1);

            var action = output.IndexOfMax();

            return new ActionInput
            {
                Inputs = inputs.Clone(),
                Action = action
            };
        }

        private void ReinitCache(int batchSize)
        {
            if (this.input == null ||
                this.input.BatchSize != batchSize)
            {
                this.output = BuilderInstance<double>.Volume.SameAs(
                    finalLayer.InputWidth,
                    finalLayer.InputHeight,
                    finalLayer.InputDepth,
                    batchSize);
                this.input = BuilderInstance<double>.Volume.SameAs(
                    inputLayer.InputWidth,
                    inputLayer.InputHeight,
                    inputLayer.InputDepth,
                    batchSize);
            }
        }

        public void Reinforce(Path[] paths)
        {
            var netGrads = this.Net.GetParametersAndGradients();
            foreach (var png in netGrads)
                png.Gradient.Clear();

            this.BatchSize = paths.Select(p => p.Count).Sum();
            ReinitCache(this.BatchSize);

            output.Clear();
            var flatInput = input.ReShape(1, 1, -1, this.BatchSize);
            this.finalLayer.BatchRewards = new double[this.BatchSize];
            int currentBatch = 0;
            foreach (var path in paths)
            {
                if (path.Used)
                    throw new NotSupportedException();

                int startOfBatch = currentBatch;
                foreach (var action in path)
                {
                    var flatAction = action.Inputs.ReShape(1, 1, -1, 1);
                    for (var d = 0; d < flatAction.Depth; d++)
                        flatInput.Set(0, 0, d, currentBatch, flatAction.Get(0, 0, d, 0));
                    output.Set(0, 0, action.Action, currentBatch, 1.0);
                    this.finalLayer.BatchRewards[currentBatch] = action.Reward;
                    currentBatch++;
                }

                //implement reward to GO (reward_t = reward_t + ... + reward_end)
                for (var batch = currentBatch - 2; batch >= startOfBatch; batch--)
                    this.finalLayer.BatchRewards[batch] += this.finalLayer.BatchRewards[batch + 1];
                
                path.Used = true;
            }

            //apply baseline
            var average = this.finalLayer.BatchRewards.Average();
            for (var i = 0; i < this.BatchSize; i++)
                this.finalLayer.BatchRewards[i] = this.finalLayer.BatchRewards[i] - average;

            this.Forward(input);
            
            this.Backward(output);

            this.Loss = average;

            //gradient ascent!
            foreach (var grad in this.Net.GetParametersAndGradients())
                grad.Gradient.DoNegate(grad.Gradient);

            var chrono = Stopwatch.StartNew();

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / paths.Sum(p => p.Count);
        }

        public override void Train(Volume<double> x, Volume<double> y)
        {
            throw new NotSupportedException("Use Reinforce method instead!");
        }
    }
}
