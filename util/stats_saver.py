import time
import util.parser as parser

def save_to_file(max_score, games_played, frame_iterations, scores, training, start_time):
    """ TODO:  Move params to Stats object or dict"""
    session_minutes = (time.time() - start_time) / 60
    stats = "\n\nMax Score: {}\nGames Played: {}\nFrame Iterations: {}\n\nScores:\n{}\nTraining: {}\nSession Time: {:.2f} minutes\n\n" \
            .format(max_score, games_played, frame_iterations, parser.sorted_dict2str(scores), training, session_minutes) + "="*40
    f = open("statistics/stats.txt", "a")
    f.write(stats)
    f.close()
