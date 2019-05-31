using System.Collections.Generic;

namespace ConvNetSharp.Core.Layers.Single
{
    public class SoftmaxLayer : SoftmaxLayer<float>
    {
        public SoftmaxLayer(int nmClasses) : base(nmClasses)
        {
        }

        public SoftmaxLayer(Dictionary<string, object> data) : base(data)
        {
        }
    }
}