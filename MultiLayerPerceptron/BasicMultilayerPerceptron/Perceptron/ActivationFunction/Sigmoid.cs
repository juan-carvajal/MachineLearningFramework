using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ActivationFunction
{
    class Sigmoid : ActivationFunction
    {
        public override double GetDerivativeValue(double x)
        {
            var result= Math.Exp(-x) / (Math.Pow(1 + Math.Exp(-x), 2));
            if (Double.IsNaN(result))
            {
                result = 0;
            }
            return result;
        }

        public override double GetValue(double x)
        {
            return 1.0 / (1 + Math.Exp(-x));
        }
    }
}
