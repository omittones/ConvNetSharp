using ConvNetSharp.Volume;
using System;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    public class ActorCriticTrainer : PolicyGradientBaseTrainer
    {
        public double Gamma { get; set; }
        public bool Bootstraping { get; set; }

        private readonly VolumeBuilder<double> builder;
        private readonly TrainerBase<double> valueFunctionTrainer;
        private Random rnd;

        public ActorCriticTrainer(
            Net<double> policyFunction,
            TrainerBase<double> valueFunctionTrainer) : base(policyFunction)
        {
            this.rnd = new Random();
            this.builder = BuilderInstance<double>.Volume;
            this.valueFunctionTrainer = valueFunctionTrainer;
            this.LearningRate = 0.1;
            this.Gamma = 0.99;
        }

        public override ActionInput Act(Volume<double> inputs)
        {
            if (Bootstraping)
            {
                return new ActionInput
                {
                    Inputs = inputs.Clone(),
                    Action = rnd.Next(0, 3)
                };
            }
            else
            {
                return base.Act(inputs);
            }
        }
        
        private Volume<double> BatchStates(Path path)
        {
            var shape = Shape.From(path.First().State.Shape);
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

        //TODO - optimize this, it does two forwards
        protected override double[] GetGradientMultipliers(Path[] paths)
        {
            var states = paths.Select(BatchStates).ToArray();
            var batchSize = states.Sum(s => s.BatchSize);

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
                    var now = value.Get(0, 0, 0, ai);
                    double error;
                    if (ai + 1 < path.Count)
                        error = path[ai].Reward + Gamma * value.Get(0, 0, 0, ai + 1) - now;
                    else
                        error = 0;

                    //if (error > 0.1)
                    //    error = 0.1;
                    //else if (error < -0.1)
                    //    error = -0.1;

                    value.Set(0, 0, 0, ai, now + error);
                }
            }
            for (var si = 0; si < states.Length; si++)
            {
                valueFunctionTrainer.Train(states[si], values[si]);
            }
            
            var advantages = new double[batchSize];

            if (!this.Bootstraping)
            {
                values = states
                    .Select(s => this.valueFunctionTrainer.Net.Forward(s, false).Clone())
                    .ToArray();

                //SetAdvantagesNotAsBaseline(paths, states, advantages);

                SetAdvantagesAsBaseline(paths, states, values, advantages);
            }

            return advantages;

            //reinforce policy
        }

        private void SetAdvantagesAsBaseline(
            Path[] paths, 
            Volume<double>[] states,
            Volume<double>[] values,
            double[] result)
        {
            //set advantages A_st_at = Expectation(R) - V_st
            int startOfBatch = 0;
            for (var pi = 0; pi < paths.Length; pi++)
            {
                var state = states[pi];
                var path = paths[pi];
                var value = values[pi];
                var advantages = new double[path.Count];
                
                for (var ai = 0; ai < advantages.Length; ai++)
                    advantages[ai] = path[ai].Reward;
                for (var ai = advantages.Length - 2; ai >= 0; ai--)
                    advantages[ai] += Gamma * advantages[ai + 1];
                for (var ai = 0; ai < advantages.Length; ai++)
                    advantages[ai] -= value.Get(0, 0, 0, ai);

                //normalize
                //var avg = advantages.Average();
                //var stdev = advantages.Select(a => (a - avg) * (a - avg)).Average();
                //stdev = Math.Sqrt(stdev);
                //for (var ai = 0; ai < advantages.Length; ai++)
                //    advantages[ai] = stdev == 0 ? 1 : (advantages[ai] - avg) / stdev;

                advantages.CopyTo(result, startOfBatch);

                startOfBatch += advantages.Length;
            }
        }

        private void SetAdvantagesNotAsBaseline(Path[] paths, Volume<double>[] states, double[] advantages)
        {
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

            //normalize
            var avg = advantages.Average();
            var stdev = advantages.Select(a => (a - avg) * (a - avg)).Average();
            stdev = Math.Sqrt(stdev);
            for (var ai = 0; ai < advantages.Length; ai++)
                advantages[ai] = stdev == 0 ? 1 : advantages[ai] / stdev;
        }
    }
}