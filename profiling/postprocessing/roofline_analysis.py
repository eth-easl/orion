import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--ai_threshold', type=float, default=9.72,
                        help='arithmetic intensity that seperates compute from memory bound kernels')
args = parser.parse_args()

df_raw = pd.read_csv(f'{args.results_dir}/raw_ncu.csv')

startp = 0
df_raw = df_raw.iloc[startp:]

l = list(df_raw.iloc[0])
print(l)
df_basic = pd.read_csv(f'{args.results_dir}/output_ncu_sms.csv', index_col=0)


dram_throughput = df_basic['DRAM_Throughput(%)']
comp_throughput = df_basic['Compute(SM)(%)']

fadd = 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'
fmul = 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'
ffma = 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed [inst/cycle]'
cycles_sec = 'smsp__cycles_elapsed.avg.per_second [cycle/nsecond]'
bytes_sec = 'dram__bytes.sum.per_second [Gbyte/second]'

ai_list = []
roofline_prof = [] # 1: comp, 0: mem, -1: invalid

comp_bound = 0
mem_bound = 0
rest = 0

for index, row in df_raw.iterrows():
    add = str(row[fadd])
    mul = str(row[fmul])
    fma = row[ffma]
    cycles = row[cycles_sec]
    bytes = row[bytes_sec]
    #print(add, mul, fma, cycles, bytes)

    if not isinstance(fma, float):
        fma = float(fma.replace("'", ''))
    add = float(add.replace("'", ''))
    mul = float(mul.replace("'", ''))


    if add or mul or fma:
        flops_cycle = add+mul+fma*2
        flops_sec = flops_cycle * cycles
        ai = flops_sec/bytes
        ai_list.append(ai)
        print(index, ai)
        if ai > args.ai_threshold:
            roofline_prof.append(1)
            comp_bound += 1
        else:
            roofline_prof.append(0)
            mem_bound += 1
    else:
        ai_list.append(0.0)
        if comp_throughput[index-startp] >= 60.0:
            roofline_prof.append(1)
        elif dram_throughput[index-startp] >= 60.0:
            roofline_prof.append(0)
        else:
            roofline_prof.append(-1)
        rest += 1


print(df_basic)
df_basic['AI(flops/bytes)'] = ai_list
df_basic['Roofline_prof'] = roofline_prof
df_basic.to_csv(f'{args.results_dir}/output_ncu_sms_roofline.csv')

print(f"comp bound: {comp_bound}, mem bound: {mem_bound}, rest: {rest}, total: {comp_bound+mem_bound+rest}")
