BIN=$1

SIZEOF_DOUBLE=8

SCALARS=(0.01 0.1 1 10 100 1000 2000 3000)

i=0
for value in ${SCALARS[*]}; do
    SIZE_BYTE[$i]=$(echo "${SCALARS[$i]}*1024*1024" | bc)
    SIZE_DOUBLES[$i]=$(echo "${SIZE_BYTE[$i]}/8" | bc)
    ((i++))
done

i=0
for value in ${SCALARS[*]}; do
    echo ${SCALARS[$i]} MB
    $BIN ${SIZE_DOUBLES[$i]}
    ((i++))
done
