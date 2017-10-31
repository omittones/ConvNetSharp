using System;

namespace ConvNetSharp.Core.Training.Double
{
    public class ReinforcementTrainer : ReinforcementTrainer<double>
    {
        public ReinforcementTrainer(
            Net<double> net,
            Random rnd) : base(net, rnd)
        {
        }
    }
}