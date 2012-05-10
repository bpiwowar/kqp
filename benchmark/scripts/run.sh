#!/bin/bash

KQP_BM="$1"
OUTDIR="$2"
test -x "$KQP_BM" || (echo "First argument should be a path to the kqp-benchmark executable" && exit 1)
test -d "$OUTDIR" || (echo "Second argument should be a path to the output directory" && exit 1)

for KEVD in direct accumulator incremental; do
for LC in "" "--no-lc"; do
while read DIM UPDATES RANKRESET RANKMAX PREIMRESET PREIMMAX; do
  # Sets some values according to the Kernel EVD algorithm
  test "$KEVD" == "direct" && LC=""
  test "$LC" == "" && PREIMRESET=$PREIMMAX
  if test "$KEVD" != "incremental"; then
    RANKRESET=$RANKMAX
  fi

  # Run the benchmark if needed
  id="$KEVD$LC-$DIM-$UPDATES-$RANKRESET-$RANKMAX-$PREIMRESET-$PREIMMAX"  
  if ! test -s "$OUTDIR/$id.out"; then
      echo "# Running $KEVD$LC ($DIM/$UPDATES) ranks=($RANKRESET,$RANKMAX) pre-images=($PREIMRESET,$PREIMMAX) [$id]"
      
      cmd=($KQP_BM kernel-evd --kevd $KEVD --updates $UPDATES --dimension $DIM --ranks $RANKRESET $RANKMAX  --pre-image-ratios $PREIMRESET $PREIMMAX $LC)
      echo "${cmd[@]}" > "$OUTDIR/$id.err"
      "${cmd[@]}" > "$OUTDIR/$id.out" 2>> "$OUTDIR/$id.err"

      if test "$?" -eq 0; then
          echo "  ... OK ..."
      else
          echo "  !!! FAILED !!!"
          rm "$OUTDIR/$id.out"
      fi
   fi 
done << EOF
100   100      80    120      1.2  2 
1000  1000     80    120      1.2  2
2000  2000     80    120      1.2  2
EOF

done # LC
done # KEVD

