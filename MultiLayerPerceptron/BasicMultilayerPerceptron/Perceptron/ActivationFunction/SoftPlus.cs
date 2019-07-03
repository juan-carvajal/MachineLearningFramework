using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ActivationFunction
{
    class SoftPlus : ActivationFunction
    {
        public override double GetDerivativeValue(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public override double GetValue(double x)
        {
            double val=Math.Log(1 + Math.Exp(x));
            if (Double.IsInfinity(val))
            {
                val = x;
            }
            return val;
        }
    }
}
