from datetime import datetime
import os
import socket
import subprocess
import time

#import beam_integrals as bi
#from beam_integrals.beam_types import BaseBeamType
#from beam_integrals.integrals import BaseIntegral
from celery import chain, chord
from celery.exceptions import Reject
import numpy as np
#import tables as tb
import csv

from .worker import simulate_pendulum_instance
from ..app import app
#from .worker import compute_integral, combine_computed_integrals_into_a_table


## Monitoring tasks

@app.task
def monitor_queues(ignore_result=True):
    server_name = app.conf.MONITORING_SERVER_NAME
    server_port = app.conf.MONITORING_SERVER_PORT
    metric_prefix = app.conf.MONITORING_METRIC_PREFIX

    queues_to_monitor = ('server', 'worker')
    
    output = subprocess.check_output('rabbitmqctl -q list_queues name messages consumers', shell=True)
    lines = (line.split() for line in output.splitlines())
    data = ((queue, int(tasks), int(consumers)) for queue, tasks, consumers in lines if queue in queues_to_monitor)

    timestamp = int(time.time())
    metrics = []
    for queue, tasks, consumers in data:
        metric_base_name = "%s.queue.%s." % (metric_prefix, queue)

        metrics.append("%s %d %d\n" % (metric_base_name + 'tasks', tasks, timestamp))
        metrics.append("%s %d %d\n" % (metric_base_name + 'consumers', consumers, timestamp))

    sock = socket.create_connection((server_name, server_port), timeout=10)
    sock.sendall(''.join(metrics))
    sock.close()


## Recording the experiment status
#ako je nesto vec racunao, on radi neke optimizacije
def get_experiment_status_filename(status):
    return os.path.join(app.conf.STATUS_DIR, status)

def get_experiment_status_time():
    """Get the current local date and time, in ISO 8601 format (microseconds and TZ removed)"""
    return datetime.now().replace(microsecond=0).isoformat()

@app.task
def record_experiment_status(status):
    with open(get_experiment_status_filename(status), 'w') as fp:
        fp.write(get_experiment_status_time() + '\n')


## Seeding the computations

def parametric_sweep(theta_resolution, tmax, dt):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    #tmax, dt = 30.0, 0.01

    theta1_inits = np.linspace(0, 2*np.pi, theta_resolution)
    theta2_inits = np.linspace(0, 2*np.pi, theta_resolution)

    import itertools
    t1t2_inits = itertools.product(theta1_inits, theta2_inits)
    return ((L1, L2, m1, m2, tmax, dt, t1t2_i[0], t1t2_i[1]) for t1t2_i in t1t2_inits)

@app.task
def seed_computations(ignore_result=True):
    #if os.path.exists(get_experiment_status_filename('started')):
        #raise Reject('Computations have already been seeded!')

    record_experiment_status.si('started').delay()
    
    _tmax = app.conf.TMAX
    _theta_res = app.conf.THETA_RESOLUTION
    _dt = app.conf.DT

    chord(
        (
            simulate_pendulum_instance.s(L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init)
            for (L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init) in
            parametric_sweep(_theta_res, _tmax, _dt)
        ),
        store_computed_integral_tables.s()
    ).delay()

def double_pendulum(beam_type_id):
    # OPTIMIZATION: Compute only the canonical integrals, as per the section 3.3.1 of my PhD thesis
    canonical_integrals = (integral for integral in BaseIntegral.plugins.instances_sorted_by_id if not integral.has_parent())

    return chord(
        (compute_integral_table(integral.id, beam_type_id) for integral in canonical_integrals),
        store_computed_integral_tables.s(beam_type_id)
    )

def compute_integral_table(integral_id, beam_type_id):
    integral = BaseIntegral.coerce(integral_id)
    max_mode = app.conf.BEAM_INTEGRALS_MAX_MODE

    # OPTIMIZATION: Don't compute integrals if their modes are equivalent, as per the section 3.3.2 of my PhD thesis
    cache_keys_seen = set()
    unique_integration_variables = []
    for variables in integral.iterate_over_used_variables(max_mode=max_mode):
        cache_key = integral.cache_key(*variables, max_mode=max_mode)
        if cache_key not in cache_keys_seen: # Cache miss
            cache_keys_seen.add(cache_key)
            unique_integration_variables.append(variables)

    return chord(
        (compute_integral.s(integral.id, beam_type_id, *variables) for variables in unique_integration_variables),
        combine_computed_integrals_into_a_table.s(integral_id)
    )


## Storing the computed integral tables

def get_hdf5_table_description(used_variables, decimal_precision):
    columns = dict(
        (var, tb.UInt8Col(pos=idx))
        for idx, var in enumerate(used_variables)
    )
    
    data_start_pos = len(used_variables)
    columns['integral_float64'] = tb.Float64Col(pos=data_start_pos)
    columns['error_float64'] = tb.Float64Col(pos=data_start_pos+1)

    columns['scale_factor'] = tb.Int8Col(pos=data_start_pos+2)
    
    max_len = decimal_precision + 10 # Account for decimal dot and exponent info
    columns['integral_str'] = tb.StringCol(itemsize=max_len, pos=data_start_pos+3)
    columns['error_str'] = tb.StringCol(itemsize=max_len, pos=data_start_pos+4)

    return columns

@app.task
def store_computed_integral_tables(solutions):    
    with open('/home/dpc.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["theta1_init", "theta2_init", "theta1_last", "theta2_last", "x1_last", "y1_last", "x2_last", "y2_last"])      
        for t1i, t2i, results in solutions:
            theta1, theta2, x1, y1, x2, y2 = results #to je ono sto je solve izracunao
            csvwriter.writerow([t1i, t2i, theta1[-1], theta2[-1], x1[-1], y1[-1], x2[-1], y2[-1]])

'''
    beam_type = BaseBeamType.coerce(beam_type_id)
    max_mode = app.conf.BEAM_INTEGRALS_MAX_MODE
    decimal_precision = app.conf.BEAM_INTEGRALS_DECIMAL_PRECISION
    hdf5_complib = app.conf.HDF5_COMPLIB
    hdf5_complevel = app.conf.HDF5_COMPLEVEL
    results_dir = app.conf.RESULTS_DIR

    # Celery `chord` sent us a list, convert it to a `dict` for easier use
    integral_tables = dict(integral_tables)

    # These filters will be applied to all the datasets created immediately under the root group:
    #   * 'complib': Specifies the compression library to be used. Although PyTables
    #     supports many interesting compression libraries, HDF5 itself provides
    #     only 2 pre-defined filters for compression: ZLIB and SZIP. We can't
    #     use SZIP due to licensing issues, therefore ZLIB has been chosen by default as
    #     it's supported by all major HDF5 viewers (HDFView, HDF Compass, ViTables,
    #     HDF Explorer).
    #   * 'complevel': Specifies a compression level for data. Using the lowest
    #     level (1) by default, per PyTables optimization recommendations (see references).
    #   * 'shuffle': Enable the Shuffle filter to improve the compression ratio.
    #   * 'fletcher32': Enable the Fletcher32 filter to add a checksum on each
    #     data chunk.
    #
    # References:
    #   * https://www.hdfgroup.org/services/filters.html
    #   * https://www.hdfgroup.org/hdf5-quest.html#gcomp
    #   * https://www.hdfgroup.org/HDF5/faq/compression.html
    #   * http://www.pytables.org/usersguide/libref/helper_classes.html#the-filters-class
    #   * http://www.pytables.org/usersguide/optimization.html#compression-issues
    #   * http://www.pytables.org/usersguide/optimization.html#shuffling-or-how-to-make-the-compression-process-more-effective
    filters = tb.Filters(complib=hdf5_complib, complevel=hdf5_complevel, shuffle=True, fletcher32=True)

    output_filename = os.path.join(results_dir, "%s.hdf5" % beam_type.filename)
    with tb.open_file(output_filename, 'w', filters=filters) as out:
        # Add the root group metadata, as per the section 3.2 of my PhD thesis
        out.root._v_attrs.created_at = get_experiment_status_time()
        out.root._v_attrs.generator_name = 'double_pendulum'
        out.root._v_attrs.generator_version = ebi.__version__
        out.root._v_attrs.beam_integrals_version = bi.__version__
        out.root._v_attrs.decimal_precision = decimal_precision
        out.root._v_attrs.max_mode = max_mode
        out.root._v_attrs.beam_type_name = beam_type.name
        out.root._v_attrs.beam_type_id = beam_type.id

        for integral in BaseIntegral.plugins.instances_sorted_by_id:
            if integral.has_parent(): 
                # OPTIMIZATION: This integral is the same as his parent, just create a hard link,
                # as per the section 3.3.1 of my PhD thesis
                parent = BaseIntegral.plugins.id_to_instance[integral.parent_id()]
                out.create_hard_link(
                    where='/',
                    name=integral.name,
                    target='/' + parent.name
                )

                # Create a hard link to the parent's index as well
                out.create_hard_link(
                    where='/',
                    name='_i_' + integral.name,
                    target='/_i_' + parent.name
                )

                continue # No further processing needed, skip to the next integral

            # Help PyTables determine the optimal chunk size
            num_rows = max_mode ** len(integral.used_variables)

            table = out.create_table(
                where='/',
                name=integral.name,
                description=get_hdf5_table_description(integral.used_variables, decimal_precision),
                expectedrows=num_rows
            )

            # Add the integral table metadata, as per the section 3.2 of my PhD thesis
            table.attrs.used_variables_list = np.array(integral.used_variables, dtype=str)
            table.attrs.used_variables_num = len(integral.used_variables)

            # OPTIMIZATION: Rationalize the number of integration variables, as per the section 3.3.3 of my PhD thesis
            row = table.row
            for m, t, v, n in integral.iterate_over_used_variables(max_mode=max_mode):
                # Store the values of `used_variables`
                d = locals()
                for var in integral.used_variables:
                    row[var] = d[var]

                # Store the integral and error data
                cache_key = integral.cache_key(m, t, v, n, max_mode=max_mode)
                data = integral_tables[integral.id][cache_key]
                for k, v in data.items():
                    row[k] = v

                row.append()

            # Flush the table manually, or the last data chunk won't be filled with correct values
            table.flush()

            # Create a completely sorted index (CSI) on all `used_variables` columns
            for var in integral.used_variables:
                table.cols._f_col(var).create_csindex()

            table.close()
'''
