# Setting up a remote jupyter session 

1. In remote machine write: `jupyter notebook --no-browser --port=8889`

2. In the local machine write:  `ssh -N -L localhost:8888:localhost:8889 username@remote.machine`

3. Access jupyter seession with a browser: `http://localhost:8888`

Now it should work. 

*PLEASE NOTE: Never commit `.ipyb` files without clearing them to the github repo.*
