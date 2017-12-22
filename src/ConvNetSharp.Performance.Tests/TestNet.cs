using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Performance.Tests
{
    public class TestNet : Net<double>
    {
        public Shape[] InputShape { get; set; }
        public Shape OutputShape { get; set; }
    }
}
