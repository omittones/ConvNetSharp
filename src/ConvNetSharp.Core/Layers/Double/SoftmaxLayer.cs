using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Double
{
    public class SoftmaxLayer : SoftmaxLayer<double>
    {
        public SoftmaxLayer(int nmClasses) : base(nmClasses)
        {
        }

        public SoftmaxLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}