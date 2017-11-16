using ConvNetSharp.Volume;
using System;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    public class ActorCriticTrainer : PolicyGradientBaseTrainer
    {
        public double Gamma { get; set; }

        private readonly VolumeBuilder<double> builder;
        private readonly TrainerBase<double> valueFunctionTrainer;

        public ActorCriticTrainer(
            Net<double> policyFunction,
            TrainerBase<double> valueFunctionTrainer) : base(policyFunction)
        {
            this.builder = BuilderInstance<double>.Volume;
            this.valueFunctionTrainer = valueFunctionTrainer;
            this.Gamma = 0.99;
            this.LearningRate = 0.1;
        }

        private Volume<double> PathState(Path path)
        {
            var shape = path.First().State.Shape;
            if (shape.GetDimension(3) != 1)
                throw new NotSupportedException();
            shape.SetDimension(3, path.Count);
            var volume = builder.SameAs(shape);

            for (var ai = 0; ai < path.Count; ai++)
                path[ai].State.CopyTo(volume, ai);

            return volume;
        }

        public Volume<double> ValueFunction(Volume<double> states)
        {
            return this.valueFunctionTrainer.Net.Forward(states, false);
        }

        protected override double[] GetGradientMultipliers(Path[] paths)
        {
            var states = paths.Select(PathState).ToArray();

            //TODO - optimize this, it does two forwards
            var values = states
                .Select(s => this.valueFunctionTrainer.Net.Forward(s, false).Clone())
                .ToArray();

            //fit valueFunction V_st = R_st + Gamma * V_st1
            for (var pi = 0; pi < states.Length; pi++)
            {
                var path = paths[pi];
                var state = states[pi];
                var value = values[pi];
                for (var ai = 0; ai < path.Count; ai++)
                {
                    double nextValue;
                    if (ai + 1 < path.Count)
                        nextValue = value.Get(0, 0, 0, ai + 1);
                    else
                        nextValue = value.Get(0, 0, 0, ai);
                    value.Set(0, 0, 0, ai, path[ai].Reward + Gamma * nextValue);
                }

                valueFunctionTrainer.Train(state, value);
            }

            var batchSize = states.Sum(s => s.BatchSize);
            var advantages = new double[batchSize];

            //set advantages A_st_at = R_st_at + Gamma * V_st1 - V_st
            int currentBatch = 0;
            for (var pi = 0; pi < paths.Length; pi++)
            {
                var state = states[pi];
                var path = paths[pi];
                var value = this.valueFunctionTrainer.Net.Forward(state, false);
                for (var ai = 0; ai < state.BatchSize; ai++)
                {
                    if (ai + 1 < state.BatchSize)
                        advantages[currentBatch] = path[ai].Reward + Gamma * value.Get(0, 0, 0, ai + 1) - value.Get(0, 0, 0, ai);
                    else
                        advantages[currentBatch] = 0;
                    currentBatch++;
                }
            }

            return advantages;

            //reinforce policy
        }
    }
}