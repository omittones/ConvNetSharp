using System;

namespace ConvNetSharp.Core.Training.Single
{
    public class ReinforcementTrainer : ReinforcementTrainer<float>
    {
        public ReinforcementTrainer(Net<float> net, Random rnd) : base(net, rnd)
        {
        }
    }
}