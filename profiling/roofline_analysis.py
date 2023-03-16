import pandas as pd
import sys

pwd = sys.argv[1]

df_raw = pd.read_csv(f'{pwd}/raw_ncu.csv')
df_basic = pd.read_csv(f'{pwd}/output_ncu_sms.csv', index_col=0)

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
    add = row[fadd]
    mul = row[fmul]
    fma = row[ffma]
    cycles = row[cycles_sec]
    bytes = row[bytes_sec]

    fma = float(fma.replace("'", ''))

    print(add, mul, fma, cycles, bytes)

    if add or mul or fma:
        flops_cycle = add+mul+fma*2
        flops_sec = flops_cycle * cycles
        ai = flops_sec/bytes
        ai_list.append(ai)
        print(ai)
        if ai > 14.94:
            roofline_prof.append(1)
            comp_bound += 1
        else:
            roofline_prof.append(0)
            mem_bound += 1
    else:
        ai_list.append(0.0)
        roofline_prof.append(-1)
        rest += 1


print(df_basic)
df_basic['AI(flops/bytes)'] = ai_list
df_basic['Roofline_prof'] = roofline_prof
df_basic.to_csv(f'{pwd}/output_ncu_sms_roofline.csv')

print(f"comp bound: {comp_bound}, mem bound: {mem_bound}, rest: {rest}")
