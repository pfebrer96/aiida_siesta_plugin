import itertools
import numpy as np

import aiida
from aiida.plugins import DataFactory
from aiida.common import AttributeDict
from aiida.engine import WorkChain, calcfunction, ToContext, while_
from aiida.orm import Float, Int, ArrayData, List, Str
from aiida.orm.utils import load_node
from aiida_siesta.calculations.tkdict import FDFDict
from aiida_siesta.workflows.base import SiestaBaseWorkChain

@calcfunction
def prepare_elastic_axis_results(pressures, structures):

    return {
        f'pressure': struct for pressure, struct in zip(pressures, structures)
    }

class ElasticAxisResponse(WorkChain):
    '''
    Computes the response of the structure to stress along an axis.

    It works by iterating over target pressures. At each target pressure,
    the structure will be relaxed allowing the cell to vary until it meets
    the requirements.

    At each iteration, it modifies:
        MD.TargetPressure
    
    It imposes values for:
        SCFMustconverge (True)
        MD.TypeOfRun ('CG')
        MD.NumCGSteps (1000)
        MD.VariableCell (True)
    '''

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # The workflow will iterate over stress steps until the structure
        # can't converge anymore or we reach a user-imposed limit.
        spec.outline(
            cls.initialize,
            while_(cls.next_step)(
                cls.run_calc,
                cls.process_outputs
            ),
            cls.return_results
        )

        # Define the inputs
        spec.input(
            'axis',
            valid_type=(Str, Int, List),
            help='''The axis that we want to compute the elastic response for.
            It can be an integer, to refer to a cell axis or a vector (List) to
            indicate a direction. E.g. tipically in an orthogonal cell 0 will be
            the same as [1,0,0].
            This input also accepts {'x', 'y', 'z'}'''
        )
        spec.input(
            'init_pressure',
            valid_type=Float,
            required=False,
            help="The initial pressure for the axis (in GPa). If not provided, will be set to 0"
        )
        spec.input(
            'pressure_step',
            valid_type=Float,
            required=False,
            help="The pressure step to take at each iteration (in GPa). Can be negative, to denote compression."
        )
        spec.input(
            'steps',
            valid_type=Int,
            required=False,
            help="The number of steps to take. If not provided, the workflow won't stop until the structure has convergence problems."
        )
        spec.input(
            'pressure_values',
            valid_type=ArrayData,
            required=False,
            help="If you don't want evenly spaced steps, you can pass the values yourself here."
        )

        spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
        spec.inputs._ports['pseudos'].dynamic = True  #Temporary fix to issue #135 plumpy
        
        #spec.output_namespace('structures', dynamic=True)
        spec.outputs.dynamic = True
        # spec.output('pressure')
        # The output will consist of a list of all the attempted values and the resulting structures.
        #spec.output('pressures', valid_type=List, help='All the calculated pressure values')
        #spec.output('structures', valid_type=List, help='The structures that have come out of each pressure value')

    def initialize(self):

        self.ctx.pressures = []
        self.ctx.structures = []

        # Normalize the axis input
        axis = self.inputs.axis
        if isinstance(axis, Str):
            axis = {'x': [1,0,0], 'y':[0,1,0], 'z':[0,0,1]}[axis.value]
        elif isinstance(axis, Int):
            axis = self.current_structure.cell[axis.value]
        elif isinstance(axis, List):
            axis = axis.get_list()

        self.ctx.axis = np.array(axis) / np.linalg.norm(axis)

        self.report(f'Starting elastic response workflow for axis {axis}')

        # The values to try will be used as iterators
        if 'init_pressure' in self.inputs:
            init_val = self.inputs.init_pressure.value
            step = self.inputs.pressure_step.value
            if 'steps' in self.inputs:
                steps = self.inputs.steps.value
                iterable = iter(np.linspace(init_val, step*(steps-1), steps))
            else:
                iterable = itertools.count(init_val, step)
        else:
            iterable = iter(self.inputs.pressure_values.get_list())

        self.ctx.values_iterable = iterable

    def next_step(self):

        # If the simulation ended without converging stop iterating
        if self.failed_simulation:
            return False
        
        # If not, proceed to put the next step value to context
        try:
            self.ctx.pressures.append(next(self.ctx.values_iterable))
        except StopIteration:
            # However, it's possible that there are no more values to try
            return False

        return True

    @property
    def failed_simulation(self):
        return getattr(self.ctx,'simulation_not_converged', False)

    @property
    def current_structure(self):
        if len(self.ctx.structures) == 0:
            structure = self.exposed_inputs(SiestaBaseWorkChain).structure
        else:
            structure = self.ctx.structures[-1]
        return structure

    def run_calc(self):
        '''
        Runs the calculation with the current value of the variable parameter.
        '''

        # Get the general inputs for the siesta run
        inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))

        # Pass the input structure for this iteration
        inputs.structure = self.current_structure
        
        # Convert the input parameters to an fdf dict to avoid duplicate keys
        parameters = FDFDict(inputs.parameters.get_dict())

        # Set the target pressure and the target stress for this run
        pressure_val = f'{self.ctx.pressures[-1]} GPa'
        parameters['md-targetpressure'] = pressure_val

        # Set also the target stress to indicate the direction of the stress
        parameters['%block mdtargetstress'] = f"""
        {" ".join( (-self.ctx.axis).astype(str))} 0 0 0
    %endblock mdtargetstress"""

        # Some other parameters
        parameters['scf-mustconverge'] = True
        parameters['md-typeofrun'] = 'CG'
        parameters['md-numcgsteps'] = 1000
        parameters['md-variablecell'] = True

        # And then just translate it again to a dict to use it for parameters
        inputs.parameters = DataFactory('dict')(dict={key: val for key, (val, _) in parameters._storage.items()})

        # Run the SIESTA simulation and store the results
        calculation = self.submit(SiestaBaseWorkChain, **inputs)

        return ToContext(calculation=calculation)

    def process_outputs(self):

        if not hasattr(self.ctx.calculation, 'outputs'):
            self.ctx.simulation_not_converged = True
            return

        outputs = self.ctx.calculation.outputs

        structure = outputs.output_structure

        self.ctx.structures.append(structure)

    def return_results(self):

        self.report(f'Elastic response computed for axis {self.ctx.axis}')

        pass

class ElasticResponse(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.outline(
            cls.submit_all,
            cls.return_results
        )

        spec.input(
            'axes',
            valid_type=List,
            default=lambda: List(list=['x', 'y', 'z']),
            help='All the axes you want to compute the elastic response for. See the axis input in ElasticAxisResponse'
        )
        spec.expose_inputs(ElasticAxisResponse, exclude=('axis',))

    def submit_all(self):

        elastic_calcs = {}

        for axis in self.inputs.axes.get_list():

            inputs = self.exposed_inputs(ElasticAxisResponse)

            inputs['axis'] = Str(axis)

            elastic_calcs[axis] = self.submit(ElasticAxisResponse, **inputs)

        return ToContext(**elastic_calcs)

    def return_results(self):
        return



    