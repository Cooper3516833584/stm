Host myd-ld25x
    HostName 192.168.31.199
    User root
    RemoteCommand none

Development and deployment rules

- Do not modify project source files on the development board directly over SSH, including with `cp`, `scp`, editors, heredocs, or generated patches.
- All source-code changes must be made and tracked in Git in the local/cloud repository first.
- Update the development-board checkout only through Git operations after the corresponding commits are available to it.
- The development board may fetch/pull approved commits, but must never push commits or branches to any remote.
- Unless the user explicitly requests it for the current task, do not run `git pull` on the development board.
