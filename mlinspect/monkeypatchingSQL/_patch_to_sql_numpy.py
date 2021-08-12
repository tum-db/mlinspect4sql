"""
Monkey patching for numpy
"""

import numpy
import gorilla
from mlinspect.monkeypatching._patch_numpy import MlinspectNdarray


@gorilla.patches(MlinspectNdarray)
class PandasPatchingSQL:
    """ Patches for pandas """

    @gorilla.name('ravel')
    @gorilla.settings(allow_hit=True)
    def patched_ravel(self, *args, **kwargs):
        """ Patch for ('pandas.io.parsers', 'read_csv') """
        # pylint: disable=no-method-argument
        return self  # We need to do the correct mapping -> dims are handled there.
