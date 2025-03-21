@files=split(/\s+/,`ls data_*.py`);
`mkdir log out json`;
foreach $f (@files)
{
 print "process $f, wait couple minutes...\n";
 `rm -f data.py`;
 `ln -s $f data.py`;

 `python test_at_dataset.py 2>&1 1> log/$f-LLHDIST.log`;
 `mv parabolic_fit.png out/parabolic_fit-$f-LLHDIST.png`;
 `mv res.png out/res-$f-LLHDIST.png`;
 `mv dec.png out/dec-$f-LLHDIST.png`;
 `mv bicfile.png out/bicfile-$f-LLHDIST.png`;
 `mkdir json/$f`;
 `mv *.json json/$f/`;
## exit(1);
}
