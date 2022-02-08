// Connect to the WebSocket Server
socket = io.connect('http://' + window.location.hostname + ':9090');

socket.on('connect', function () {
    console.log('Client has connected to the server!');
    stop_attack();
});

socket.on('disconnect', function () {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

function clear_patch() {
    socket.emit('clear_patch', 1);
}

function start_attack() {
    socket.emit('fix_patch', 0);
    $('#start').hide();
    $('#pause').show();
}

function pause_attack() {
    socket.emit('fix_patch', 1);
    $('#start').show();
    $('#pause').hide();
}

function stop_attack() {
    socket.emit('fix_patch', 1);
    socket.emit('clear_patch', 1);
}

function save_patch() {
    socket.emit('save_patch');
    new simpleSnackbar('Filter Saved', {
        type: 'info',
    }).show();
}

$(document).ready(function () {
    $('#pause').hide();
});
