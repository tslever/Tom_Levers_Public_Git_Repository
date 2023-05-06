# Tom_Levers_Public_Git_Repository
Contains Tom Lever's public software solutions

To resolve "Filename too long" in Windows 10, run `git config --global core.longpaths true`.

To merge the development branch of Git repository ~/B into the development branch of Git repository ~/A, from ~ run:

cd A

git remote add B ../B

git fetch B

git merge B/development --allow-unrelated

git remote remove B

See https://stackoverflow.com/questions/56051560/how-to-merge-two-git-repositories-without-loosing-the-history-for-either-reposit .