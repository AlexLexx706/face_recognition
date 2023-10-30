# Sevice commands:
## checks service log:
`journalctl -u sneak.service`

## stop service:
`systemctl stop sneak.service`

## start service:
`systemctl start sneak.service`

## enable/disable and reload service
`systemctl enable sneak.service`
`systemctl disable sneak.service`
`systemctl reload sneak.service`

# service file:
`/lib/systemd/system/sneak.service`
