@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET count=1
FOR /R  C:\Users\lukas.rimkus\.keras\datasets\flower_photos\daisy\ %%G in (*.jpg) DO (call :subroutine "%%G")
GOTO :eof

:subroutine
    ECHO %count%: %1
    set /a count +=1
    IF %count% == 100 (
        ECHO "100!"
        GOTO END)
    GOTO :eof
    :END EXIT /B 0

