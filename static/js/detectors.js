$(function () {

    const detection_cmd = ['async', 'sync', 'person_det', 'person_reid', 'show_track'];

    $.ajaxSetup({ cache: false });

    // post detection action
    $('.btn').on('click', function (e) {
        var command = JSON.stringify({ "command": $('#' + $(this).attr('id')).val() });

        if (JSON.parse(command).command == "") {
            var command = JSON.stringify({ "command": $(this).find('input').val() });
        }
        //console.log("btn", command)
        if (detection_cmd.includes(JSON.parse(command).command)) {
            post('/detection', command);
        }
        if (JSON.parse(command).command == 'flip') {
            post('/flip', command);
        }
    });

    // ajax post
    function post(url, command) {
        $.ajax({
            type: 'POST',
            url: url,
            data: command,
            contentType: 'application/json',
            timeout: 10000
        }).done(function (data, textStatus) {
            let post_command = JSON.parse(command).command;
            let is_det = JSON.parse(data.ResultSet).is_det;
            let is_reid = JSON.parse(data.ResultSet).is_reid;

            //console.log("post_command", post_command);

            $("#res").text("Command: " + post_command + " Status: " + textStatus);

            if (JSON.parse(command).command == 'async') {
                $("#async").attr('class', 'btn btn-danger');
                $("#sync").attr('class', 'btn btn-dark');
            }

            if (JSON.parse(command).command == 'sync') {
                $("#sync").attr('class', 'btn btn-danger');
                $("#async").attr('class', 'btn btn-dark');
            }

            if (is_det && post_command == "person-det") {
                $('#video_feed').fadeIn(100);
                $("#person-det").attr('class', 'btn btn-secondary active');
                $("#person-reid").attr('class', 'btn btn-secondary');
            } else if (!is_det) {
                $("#person-det").attr('class', 'btn btn-secondary');
            }

            if (is_reid && post_command == "person-reid") {
                $('#video_feed').fadeIn(100);
                $("#person-det").attr('class', 'btn btn-secondary');
                $("#person-reid").attr('class', 'btn btn-secondary active');
            } else if (!is_reid) {
                $("#person-reid").attr('class', 'btn btn-secondary');
            }


        }).fail(function (jqXHR, textStatus, errorThrown) {
            $("#res").text(textStatus + ":" + jqXHR.status + " " + errorThrown);
        });
        return false;
    }
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});

