using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ActivationFunction
{
    class ReLu : ActivationFunction
    {
        public override double GetDerivativeValue(double x)
        {
            if (x <= 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public override double GetValue(double x)
        {
            return Math.Max(0, x);
        }
    }
}
